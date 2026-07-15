from __future__ import annotations

import fcntl
import json
import math
import os
import shutil
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, Sequence
from uuid import uuid4

import numpy as np

from .graph import DOCUMENT_SCHEMA, GraphifyGraph


CACHE_SCHEMA = 5


class Embedder(Protocol):
    model_name: str

    def cache_identity(self) -> dict[str, Any]: ...

    def encode(
        self, texts: Sequence[str], *, show_progress: bool = False
    ) -> np.ndarray: ...


class Reranker(Protocol):
    model_name: str

    def score(
        self, query: str, documents: Sequence[str], batch_size: int = 1
    ) -> np.ndarray: ...


class EmbeddingIndex:
    def __init__(self, graph: GraphifyGraph):
        self.graph = graph
        self.cache_dir = graph.path.parent / "cache"
        self.metadata_path = self.cache_dir / "embeddings.json"
        self.vectors_path = self.cache_dir / "embeddings.npz"
        self.metadata: dict[str, Any] = {}
        self.node_ids: list[str] = []
        self.vectors = np.empty((0, 0), dtype=np.float32)

    def exists(self) -> bool:
        return self.metadata_path.is_file() and self.vectors_path.is_file()

    def load(self) -> "EmbeddingIndex":
        if not self.exists():
            raise FileNotFoundError(
                f"Embedding index missing beside {self.graph.path}; run the index command first"
            )
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("schema_version") != CACHE_SCHEMA:
            raise ValueError("Unsupported embedding cache schema")
        with np.load(self.vectors_path, allow_pickle=False) as data:
            self.node_ids = [str(value) for value in data["node_ids"].tolist()]
            self.vectors = np.asarray(data["vectors"], dtype=np.float32)
            vector_generation = str(data["generation_id"].item())
        expected_rows = int(self.metadata.get("node_count", -1))
        expected_dimension = int(self.metadata.get("dimension", -1))
        if self.vectors.ndim != 2 or self.vectors.shape[0] != len(self.node_ids):
            raise ValueError("Embedding cache is inconsistent")
        if len(set(self.node_ids)) != len(self.node_ids):
            raise ValueError("Embedding cache contains duplicate node IDs")
        if set(self.node_ids) != set(self.graph.by_id):
            raise ValueError("Embedding cache node IDs do not match the current graph")
        content_hashes = self.metadata.get("content_hashes")
        if not isinstance(content_hashes, dict) or set(content_hashes) != set(
            self.node_ids
        ):
            raise ValueError("Embedding cache content hashes are incomplete")
        identity = self.metadata.get("embedding_identity")
        if (
            not isinstance(identity, dict)
            or identity.get("document_schema") != DOCUMENT_SCHEMA
        ):
            raise ValueError("Embedding cache text-construction schema is incompatible")
        if (
            expected_rows != len(self.node_ids)
            or expected_dimension != self.vectors.shape[1]
        ):
            raise ValueError("Embedding cache metadata dimensions do not match vectors")
        if vector_generation != str(self.metadata.get("generation_id")):
            raise ValueError("Embedding cache files belong to different generations")
        if not np.isfinite(self.vectors).all():
            raise ValueError("Embedding cache contains non-finite vectors")
        if len(self.vectors):
            norms = np.linalg.norm(self.vectors, axis=1)
            if not np.allclose(norms, 1.0, rtol=1e-3, atol=1e-3):
                raise ValueError("Embedding cache contains non-normalized vectors")
        return self

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(
                    payload, handle, indent=2, ensure_ascii=False, allow_nan=False
                )
                handle.write("\n")
            os.replace(temporary, path)
        except Exception:
            Path(temporary).unlink(missing_ok=True)
            raise

    @staticmethod
    def _write_vectors_atomic(
        path: Path, node_ids: list[str], vectors: np.ndarray, generation_id: str
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=path.name + ".", suffix=".npz", dir=path.parent
        )
        os.close(fd)
        try:
            np.savez_compressed(
                temporary,
                node_ids=np.asarray(node_ids, dtype=np.str_),
                vectors=np.asarray(vectors, dtype=np.float32),
                generation_id=np.asarray(generation_id, dtype=np.str_),
            )
            os.replace(temporary, path)
        except Exception:
            Path(temporary).unlink(missing_ok=True)
            raise

    @contextmanager
    def _write_lock(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.cache_dir / ".embeddings.lock"
        with lock_path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def build(
        self,
        embedder: Embedder,
        *,
        include_source: bool = True,
        force: bool = False,
        show_progress: bool = True,
    ) -> dict[str, Any]:
        with self._write_lock():
            return self._build_locked(
                embedder,
                include_source=include_source,
                force=force,
                show_progress=show_progress,
            )

    def _build_locked(
        self,
        embedder: Embedder,
        *,
        include_source: bool = True,
        force: bool = False,
        show_progress: bool = True,
    ) -> dict[str, Any]:
        documents = self.graph.documents(include_source=include_source)
        node_ids = [node_id for node_id, _ in documents]
        text_by_id = dict(documents)
        hashes = self.graph.content_hashes(include_source=include_source)
        identity_factory = getattr(embedder, "cache_identity", None)
        if callable(identity_factory):
            embedding_identity = dict(identity_factory())
        else:
            embedding_identity = {
                "model": embedder.model_name,
                "backend": getattr(embedder, "backend", None),
                "instruction": getattr(embedder, "instruction", None),
                "revision": getattr(embedder, "revision", None),
                "wrapper_sha256": getattr(embedder, "wrapper_sha256", None),
                "artifact_fingerprint": getattr(embedder, "artifact_fingerprint", None),
                "dtype": str(getattr(embedder, "dtype", "unknown")),
            }
        embedding_identity["document_schema"] = DOCUMENT_SCHEMA

        old_vectors: dict[str, np.ndarray] = {}
        old_hashes: dict[str, str] = {}
        if self.exists() and not force:
            try:
                self.load()
                if (
                    self.metadata.get("embedding_identity") == embedding_identity
                    and self.metadata.get("include_source") == include_source
                ):
                    old_hashes = {
                        str(key): str(value)
                        for key, value in self.metadata.get(
                            "content_hashes", {}
                        ).items()
                    }
                    old_vectors = {
                        node_id: self.vectors[index]
                        for index, node_id in enumerate(self.node_ids)
                    }
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                old_vectors = {}
                old_hashes = {}

        changed = [
            node_id
            for node_id in node_ids
            if old_hashes.get(node_id) != hashes[node_id] or node_id not in old_vectors
        ]
        new_vectors = embedder.encode(
            [text_by_id[node_id] for node_id in changed],
            show_progress=show_progress,
        )
        if changed and (new_vectors.ndim != 2 or new_vectors.shape[0] != len(changed)):
            raise RuntimeError(
                "Embedding backend returned the wrong number or shape of vectors"
            )
        if changed and not np.isfinite(new_vectors).all():
            raise RuntimeError("Embedding backend returned non-finite vectors")

        combined = dict(old_vectors)
        for index, node_id in enumerate(changed):
            combined[node_id] = new_vectors[index]
        if node_ids:
            missing = [node_id for node_id in node_ids if node_id not in combined]
            if missing:
                raise RuntimeError(f"No vectors produced for {len(missing)} node(s)")
            vectors = np.stack([combined[node_id] for node_id in node_ids]).astype(
                np.float32
            )
        else:
            vectors = np.empty((0, 0), dtype=np.float32)

        if not np.isfinite(vectors).all():
            raise RuntimeError("Combined embedding index contains non-finite vectors")
        if len(vectors):
            norms = np.linalg.norm(vectors, axis=1)
            if not np.allclose(norms, 1.0, rtol=1e-3, atol=1e-3):
                raise RuntimeError("Embedding backend returned non-normalized vectors")
        generation_id = uuid4().hex
        metadata = {
            "schema_version": CACHE_SCHEMA,
            "generation_id": generation_id,
            "model": embedder.model_name,
            "backend": getattr(embedder, "backend", None),
            "instruction": getattr(embedder, "instruction", None),
            "requested_dtype": getattr(embedder, "requested_dtype", None),
            "embedding_identity": embedding_identity,
            "dimension": int(vectors.shape[1])
            if vectors.ndim == 2 and vectors.size
            else 0,
            "node_count": len(node_ids),
            "include_source": include_source,
            "graph": str(self.graph.path),
            "graph_commit": self.graph.data.get("built_at_commit"),
            "content_hashes": hashes,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._write_vectors_atomic(self.vectors_path, node_ids, vectors, generation_id)
        self._write_json_atomic(self.metadata_path, metadata)
        self.metadata = metadata
        self.node_ids = node_ids
        self.vectors = vectors
        return {
            "nodes": len(node_ids),
            "embedded": len(changed),
            "reused": len(node_ids) - len(changed),
            "dimension": metadata["dimension"],
            "model": embedder.model_name,
            "device": getattr(embedder, "device", None),
            "metadata_path": str(self.metadata_path),
            "vectors_path": str(self.vectors_path),
        }

    def search(
        self,
        query: str,
        query_vector: np.ndarray,
        *,
        top_k: int = 10,
        candidate_k: int = 24,
        neighbors: int = 1,
        semantic_weight: float = 0.9,
        lexical_weight: float = 0.08,
        structure_weight: float = 0.02,
        reranker: Reranker | None = None,
        rerank_batch_size: int = 1,
    ) -> list[dict[str, Any]]:
        if int(top_k) < 1 or int(candidate_k) < 1:
            raise ValueError("top_k and candidate_k must be positive")
        if int(neighbors) < 0 or int(rerank_batch_size) < 1:
            raise ValueError(
                "neighbors must be non-negative and rerank_batch_size positive"
            )
        if not all(
            math.isfinite(float(weight))
            for weight in (semantic_weight, lexical_weight, structure_weight)
        ):
            raise ValueError("Search weights must be finite")
        if not self.node_ids:
            self.load()
        vector = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if not np.isfinite(vector).all():
            raise ValueError("Query embedding contains non-finite values")
        if self.vectors.shape[1] != vector.shape[0]:
            raise ValueError(
                f"Query dimension {vector.shape[0]} != index dimension {self.vectors.shape[1]}"
            )
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            raise ValueError("Query embedding has zero norm")
        semantic_scores = self.vectors @ (vector / norm)
        max_degree = max(
            (self.graph.degree(node_id) for node_id in self.node_ids), default=1
        )

        ranked: list[dict[str, Any]] = []
        text_cache: dict[str, str] = {}
        for index, node_id in enumerate(self.node_ids):
            node = self.graph.by_id.get(node_id)
            if node is None:
                continue
            text = self.graph.node_text(node, include_source=False)
            text_cache[node_id] = text
            lexical = self.graph.lexical_score(query, text)
            structural = self.graph.degree(node_id) / max(max_degree, 1)
            score = (
                semantic_weight * float(semantic_scores[index])
                + lexical_weight * lexical
                + structure_weight * structural
            )
            ranked.append(
                {
                    "id": node_id,
                    "label": node.get("label", node_id),
                    "source_file": node.get("source_file"),
                    "source_location": node.get("source_location"),
                    "community": node.get("community"),
                    "semantic_score": round(float(semantic_scores[index]), 6),
                    "lexical_score": round(float(lexical), 6),
                    "structural_score": round(float(structural), 6),
                    "retrieval_score": round(float(score), 6),
                }
            )
        ranked.sort(key=lambda item: item["retrieval_score"], reverse=True)
        pool = ranked[: max(int(candidate_k), int(top_k))]

        if reranker is not None and pool:
            rerank_scores = reranker.score(
                query,
                [text_cache[item["id"]] for item in pool],
                batch_size=rerank_batch_size,
            )
            if len(rerank_scores) != len(pool):
                raise RuntimeError("Reranker returned the wrong number of scores")
            if not np.isfinite(rerank_scores).all():
                raise RuntimeError("Reranker returned non-finite scores")
            for item, rerank_score in zip(pool, rerank_scores, strict=True):
                item["reranker_score"] = round(float(rerank_score), 6)
                item["score"] = round(
                    0.35 * float(item["retrieval_score"]) + 0.65 * float(rerank_score),
                    6,
                )
            pool.sort(key=lambda item: item["score"], reverse=True)
        else:
            for item in pool:
                item["score"] = item["retrieval_score"]

        results = pool[: int(top_k)]
        if neighbors > 0:
            for item in results:
                item["neighbors"] = self.graph.neighbors(item["id"], depth=neighbors)
        return results

    def similarity_pairs(
        self,
        *,
        threshold: float = 0.82,
        max_neighbors: int = 5,
        block_size: int = 512,
    ) -> list[tuple[str, str, float]]:
        if not self.node_ids:
            self.load()
        count = len(self.node_ids)
        threshold = float(threshold)
        if not math.isfinite(threshold) or not -1.0 <= threshold <= 1.0:
            raise ValueError("threshold must be finite and within [-1, 1]")
        if int(max_neighbors) < 1 or int(block_size) < 1:
            raise ValueError("max_neighbors and block_size must be positive")
        if count < 2:
            return []
        max_neighbors = int(max_neighbors)
        pairs: dict[tuple[str, str], float] = {}
        for start in range(0, count, int(block_size)):
            stop = min(count, start + int(block_size))
            scores = self.vectors[start:stop] @ self.vectors.T
            for local_index, row in enumerate(scores):
                source_index = start + local_index
                row[source_index] = -np.inf
                candidate_count = min(max_neighbors, count - 1)
                candidate_indices = np.argpartition(row, -candidate_count)[
                    -candidate_count:
                ]
                candidate_indices = candidate_indices[
                    np.argsort(row[candidate_indices])[::-1]
                ]
                for target_index in candidate_indices:
                    score = float(row[target_index])
                    if score < threshold:
                        continue
                    source = self.node_ids[source_index]
                    target = self.node_ids[int(target_index)]
                    key = tuple(sorted((source, target)))
                    pairs[key] = max(score, pairs.get(key, -1.0))
        return [
            (source, target, score)
            for (source, target), score in sorted(
                pairs.items(), key=lambda item: item[1], reverse=True
            )
        ]

    def write_linked_graph(
        self,
        pairs: Sequence[tuple[str, str, float]],
        *,
        in_place: bool = False,
        output: str | Path | None = None,
    ) -> tuple[Path, int]:
        model = str(self.metadata.get("model") or "unknown")
        semantic_links = self.graph.semantic_links(pairs, model=model)
        existing_nonsemantic = [
            link
            for link in self.graph.links
            if (link.get("relation") or link.get("type")) != "semantically_similar_to"
        ]
        payload = dict(self.graph.data)
        payload["links"] = existing_nonsemantic + semantic_links
        payload.setdefault("graph", {})
        if isinstance(payload["graph"], dict):
            payload["graph"]["embedding_model"] = model
            payload["graph"]["semantic_edge_count"] = len(semantic_links)

        if in_place:
            target = self.graph.path
            backup = target.with_suffix(target.suffix + ".bak")
            shutil.copy2(target, backup)
        elif output is not None:
            target = Path(output).expanduser().resolve()
            if target == self.graph.path:
                raise ValueError(
                    "--output cannot overwrite the input graph; use --in-place to create a backup"
                )
        else:
            target = self.graph.path.with_name(self.graph.path.stem + ".semantic.json")
        self._write_json_atomic(target, payload)
        return target, len(semantic_links)
