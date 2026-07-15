from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .graph import DOCUMENT_SCHEMA, GraphifyGraph
from .index import EmbeddingIndex


FileSignature = tuple[int, int, int, int, int]
IndexSignature = tuple[FileSignature, ...]


def _positive_int(payload: dict[str, Any], key: str, default: int) -> int:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _nonnegative_int(payload: dict[str, Any], key: str, default: int) -> int:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return value


class ModelService:
    def __init__(
        self,
        manager: Any,
        *,
        idle_timeout_seconds: float = 60.0,
        clock=time.monotonic,
    ):
        self.manager = manager
        if idle_timeout_seconds <= 0:
            raise ValueError("idle_timeout_seconds must be greater than zero")
        self.idle_timeout_seconds = float(idle_timeout_seconds)
        self._clock = clock
        self._resident_index: EmbeddingIndex | None = None
        self._index_deadline: float | None = None
        self._resident_signature: IndexSignature | None = None
        self._index_eviction_pending = False

    @staticmethod
    def _graph_path(payload: dict[str, Any]) -> Path:
        path = payload.get("graph")
        if not isinstance(path, str) or not path:
            raise ValueError("graph must be a non-empty path")
        return Path(path).expanduser().resolve()

    @staticmethod
    def _graph(payload: dict[str, Any]) -> GraphifyGraph:
        path = payload.get("graph")
        if not isinstance(path, str) or not path:
            raise ValueError("graph must be a non-empty path")
        return GraphifyGraph(path)

    def _resident_for(
        self,
        payload: dict[str, Any],
        current_signature: IndexSignature | None,
    ) -> EmbeddingIndex | None:
        self._expire_index()
        resident = self._resident_index
        if resident is None:
            return None
        if resident.graph.path != self._graph_path(payload):
            self._unload_index()
            return None
        if current_signature is None or current_signature != self._resident_signature:
            self._unload_index()
            return None
        return resident

    @staticmethod
    def _file_signature(path: Path) -> FileSignature | None:
        try:
            stat = path.stat()
        except FileNotFoundError:
            return None
        return (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )

    @classmethod
    def _index_signature(cls, graph_path: Path) -> IndexSignature | None:
        cache_dir = graph_path.parent / "cache"
        paths = (
            graph_path,
            cache_dir / "embeddings.json",
            cache_dir / "embeddings.npz",
        )
        signatures = []
        for path in paths:
            signature = cls._file_signature(path)
            if signature is None:
                return None
            signatures.append(signature)
        return tuple(signatures)

    @classmethod
    def _built_index_signature(
        cls,
        index: EmbeddingIndex,
        expected_graph_signature: FileSignature | None,
    ) -> IndexSignature | None:
        before = cls._index_signature(index.graph.path)
        if (
            before is None
            or expected_graph_signature is None
            or before[0] != expected_graph_signature
        ):
            return None
        try:
            metadata = json.loads(index.metadata_path.read_text(encoding="utf-8"))
            with np.load(index.vectors_path, allow_pickle=False) as data:
                vector_generation = str(data["generation_id"].item())
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return None
        after = cls._index_signature(index.graph.path)
        generation = str(index.metadata.get("generation_id"))
        if (
            after != before
            or str(metadata.get("generation_id")) != generation
            or vector_generation != generation
        ):
            return None
        return after

    def _touch_index(self) -> None:
        if self._resident_index is not None:
            self._index_deadline = self._clock() + self.idle_timeout_seconds

    def _unload_index(self) -> bool:
        if self._resident_index is None:
            return False
        self._resident_index = None
        self._index_deadline = None
        self._resident_signature = None
        self._index_eviction_pending = True
        gc.collect()
        return True

    def _expire_index(self) -> bool:
        if (
            self._resident_index is not None
            and self._index_deadline is not None
            and self._clock() >= self._index_deadline
        ):
            self._unload_index()
            return True
        return False

    def tick(self) -> list[str]:
        self._expire_index()
        if not self._index_eviction_pending:
            return []
        self._index_eviction_pending = False
        return ["index"]

    def status(self) -> dict[str, Any] | None:
        self._expire_index()
        index = self._resident_index
        if index is None:
            return None
        if self._index_signature(index.graph.path) != self._resident_signature:
            self._unload_index()
            return None
        deadline = self._index_deadline
        return {
            "graph": str(index.graph.path),
            "node_count": len(index.node_ids),
            "dimension": int(index.vectors.shape[1]) if index.vectors.ndim == 2 else 0,
            "idle_seconds_remaining": (
                max(0.0, deadline - self._clock()) if deadline is not None else 0.0
            ),
        }

    def close(self) -> None:
        self._unload_index()

    @staticmethod
    def _index_is_current(
        graph: GraphifyGraph,
        index: EmbeddingIndex,
        embedder: Any,
        *,
        include_source: bool,
        verify_content_hashes: bool = True,
    ) -> bool:
        try:
            if not index.metadata:
                index.load()
        except (OSError, ValueError, KeyError):
            return False
        identity = dict(embedder.cache_identity())
        identity["document_schema"] = DOCUMENT_SCHEMA
        return (
            index.metadata.get("embedding_identity") == identity
            and index.metadata.get("include_source") == include_source
            and (
                not verify_content_hashes
                or index.metadata.get("content_hashes")
                == graph.content_hashes(include_source=include_source)
            )
            and set(index.node_ids) == set(graph.by_id)
        )

    def index(self, payload: dict[str, Any]) -> dict[str, Any]:
        graph = self._graph(payload)
        self._unload_index()
        include_source = bool(payload.get("include_source", True))
        force = bool(payload.get("force", False))
        embedder = self.manager.get_embedder()
        return EmbeddingIndex(graph).build(
            embedder,
            include_source=include_source,
            force=force,
            show_progress=False,
        )

    def search(self, payload: dict[str, Any]) -> dict[str, Any]:
        graph_path = self._graph_path(payload)
        signature_before = self._index_signature(graph_path)
        graph_signature_before = self._file_signature(graph_path)
        index = self._resident_for(payload, signature_before)
        was_resident = index is not None
        graph = index.graph if index is not None else GraphifyGraph(graph_path)
        query = payload.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        include_source = bool(payload.get("include_source", True))
        embedder = self.manager.get_embedder()
        if index is None:
            index = EmbeddingIndex(graph)
        build_stats = None
        if bool(payload.get("force_index", False)) or not self._index_is_current(
            graph,
            index,
            embedder,
            include_source=include_source,
            verify_content_hashes=not was_resident,
        ):
            build_stats = index.build(
                embedder,
                include_source=include_source,
                force=bool(payload.get("force_index", False)),
                show_progress=False,
            )
        query_vector = embedder.encode([query], show_progress=False)[0]
        reranker = (
            self.manager.get_reranker(allow_disabled=True)
            if bool(payload.get("rerank", False))
            else None
        )
        results = index.search(
            query,
            query_vector,
            top_k=_positive_int(payload, "top_k", 10),
            candidate_k=_positive_int(payload, "candidate_k", 24),
            neighbors=_nonnegative_int(payload, "neighbors", 1),
            reranker=reranker,
            rerank_batch_size=_positive_int(payload, "reranker_batch_size", 1),
        )
        signature_after = self._index_signature(graph.path)
        signature = (
            signature_after
            if signature_before is not None and signature_after == signature_before
            else None
        )
        if signature is None and build_stats is not None:
            signature = self._built_index_signature(index, graph_signature_before)
        if signature is None:
            self._unload_index()
        else:
            self._resident_index = index
            self._resident_signature = signature
            self._touch_index()
        return {
            "query": query,
            "graph": str(graph.path),
            "embedding_model": index.metadata.get("model"),
            "reranker_model": getattr(reranker, "model_name", None),
            "index_build": build_stats,
            "results": results,
        }
