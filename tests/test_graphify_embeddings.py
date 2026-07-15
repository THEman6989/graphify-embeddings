from __future__ import annotations

import argparse
import json
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np

from graphify_embeddings import cli
from graphify_embeddings.graph import GraphifyGraph
from graphify_embeddings.index import EmbeddingIndex
from graphify_embeddings.models import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RERANKER_MODEL,
    QwenEmbedder,
    QwenReranker,
)


class FakeEmbedder:
    model_name = "fake/qwen-8b"
    device = "cuda:0"
    backend = "fake"
    revision = "test-revision"
    wrapper_sha256 = None
    requested_dtype = "fp32"

    def __init__(self, instruction="test instruction"):
        self.instruction = instruction
        self.calls: list[list[str]] = []

    def cache_identity(self):
        return {
            "model": self.model_name,
            "backend": self.backend,
            "instruction": self.instruction,
            "revision": self.revision,
            "wrapper_sha256": self.wrapper_sha256,
            "dtype": self.requested_dtype,
        }

    def encode(self, texts, *, show_progress=False):
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            lowered = text.lower()
            vector = np.array(
                [
                    lowered.count("cache") + lowered.count("embedding") + 0.1,
                    lowered.count("model") + lowered.count("gpu") + 0.1,
                    lowered.count("beat") + lowered.count("audio") + 0.1,
                    lowered.count("rerank") + lowered.count("score") + 0.1,
                ],
                dtype=np.float32,
            )
            vector /= np.linalg.norm(vector)
            vectors.append(vector)
        return np.stack(vectors) if vectors else np.empty((0, 4), dtype=np.float32)


class SlowFakeEmbedder(FakeEmbedder):
    def encode(self, texts, *, show_progress=False):
        time.sleep(0.1)
        return super().encode(texts, show_progress=show_progress)


class ClosingEmbedder(FakeEmbedder):
    def __init__(self):
        super().__init__()
        self.closed = False

    def close(self):
        self.closed = True


class FakeReranker:
    model_name = "fake/reranker-8b"

    def score(self, query, documents, batch_size=1):
        return np.asarray(
            [0.99 if "rerank" in document.lower() else 0.10 for document in documents],
            dtype=np.float32,
        )


class GraphifyEmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "cache.py").write_text(
            "def load_embedding_cache():\n    return 'cache'\n", encoding="utf-8"
        )
        (self.root / "models.py").write_text(
            "def load_gpu_model():\n    return 'model'\n", encoding="utf-8"
        )
        output = self.root / "graphify-out"
        output.mkdir()
        graph = {
            "directed": False,
            "multigraph": False,
            "graph": {},
            "nodes": [
                {
                    "id": "cache-loader",
                    "label": "load_embedding_cache",
                    "docstring": "Load vectors from the persistent embedding cache",
                    "source_file": "cache.py",
                    "source_location": "L1",
                    "community": 0,
                },
                {
                    "id": "gpu-loader",
                    "label": "load_gpu_model",
                    "docstring": "Load a Qwen model on the first NVIDIA GPU",
                    "source_file": "models.py",
                    "source_location": "L1",
                    "community": 1,
                },
                {
                    "id": "reranker",
                    "label": "rerank_candidates",
                    "docstring": "Rerank candidate documents with cross attention scores",
                    "source_file": "rank.py",
                    "source_location": "L8",
                    "community": 1,
                },
            ],
            "links": [
                {
                    "source": "gpu-loader",
                    "target": "reranker",
                    "relation": "calls",
                    "confidence": "EXTRACTED",
                }
            ],
            "hyperedges": [],
            "built_at_commit": "abc123",
        }
        self.graph_path = output / "graph.json"
        self.graph_path.write_text(json.dumps(graph), encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def test_node_text_includes_source_context_without_path_escape(self):
        graph = GraphifyGraph(self.graph_path)
        text = graph.node_text(graph.by_id["cache-loader"])
        self.assertIn("load_embedding_cache", text)
        self.assertIn("source_context", text)
        escaped = dict(graph.by_id["cache-loader"], source_file="../secret")
        self.assertEqual(graph.source_context(escaped), "")

    def test_incremental_cache_reuses_unchanged_vectors(self):
        graph = GraphifyGraph(self.graph_path)
        index = EmbeddingIndex(graph)
        embedder = FakeEmbedder()
        first = index.build(embedder, show_progress=False)
        self.assertEqual(first["embedded"], 3)
        self.assertEqual(first["reused"], 0)
        self.assertTrue(index.metadata_path.is_file())
        self.assertTrue(index.vectors_path.is_file())

        second_embedder = FakeEmbedder()
        second = index.build(second_embedder, show_progress=False)
        self.assertEqual(second["embedded"], 0)
        self.assertEqual(second["reused"], 3)
        self.assertEqual(second_embedder.calls, [[]])

        graph.data["nodes"][0]["docstring"] += " changed"
        self.graph_path.write_text(json.dumps(graph.data), encoding="utf-8")
        changed_index = EmbeddingIndex(GraphifyGraph(self.graph_path))
        third_embedder = FakeEmbedder()
        third = changed_index.build(third_embedder, show_progress=False)
        self.assertEqual(third["embedded"], 1)
        self.assertEqual(third["reused"], 2)

    def test_semantic_search_and_graph_neighbors(self):
        graph = GraphifyGraph(self.graph_path)
        index = EmbeddingIndex(graph)
        embedder = FakeEmbedder()
        index.build(embedder, show_progress=False)
        query = embedder.encode(["embedding cache"], show_progress=False)[0]
        results = index.search("embedding cache", query, top_k=2, neighbors=1)
        self.assertEqual(results[0]["id"], "cache-loader")

        gpu_query = embedder.encode(["gpu model"], show_progress=False)[0]
        gpu_results = index.search("gpu model", gpu_query, top_k=1, neighbors=1)
        self.assertEqual(gpu_results[0]["id"], "gpu-loader")
        self.assertEqual(gpu_results[0]["neighbors"][0]["id"], "reranker")
        self.assertEqual(gpu_results[0]["neighbors"][0]["relation"], "calls")

    def test_reranker_reorders_only_candidate_pool(self):
        graph = GraphifyGraph(self.graph_path)
        index = EmbeddingIndex(graph)
        embedder = FakeEmbedder()
        index.build(embedder, show_progress=False)
        query = embedder.encode(["gpu model"], show_progress=False)[0]
        results = index.search(
            "gpu model",
            query,
            top_k=2,
            candidate_k=3,
            neighbors=0,
            reranker=FakeReranker(),
        )
        self.assertEqual(results[0]["id"], "reranker")
        self.assertIn("reranker_score", results[0])

    def test_instruction_change_invalidates_all_cached_vectors(self):
        graph = GraphifyGraph(self.graph_path)
        index = EmbeddingIndex(graph)
        index.build(FakeEmbedder("instruction one"), show_progress=False)
        replacement = FakeEmbedder("instruction two")
        stats = index.build(replacement, show_progress=False)
        self.assertEqual(stats["embedded"], 3)
        self.assertEqual(stats["reused"], 0)

    def test_concurrent_builds_are_serialized_by_cache_lock(self):
        graph = GraphifyGraph(self.graph_path)

        def build_once():
            return EmbeddingIndex(graph).build(SlowFakeEmbedder(), show_progress=False)

        with ThreadPoolExecutor(max_workers=2) as pool:
            stats = list(pool.map(lambda _: build_once(), range(2)))
        self.assertEqual(sorted(item["embedded"] for item in stats), [0, 3])
        loaded = EmbeddingIndex(graph).load()
        self.assertEqual(len(loaded.node_ids), 3)

    def test_cache_generation_and_finite_vector_validation(self):
        graph = GraphifyGraph(self.graph_path)
        index = EmbeddingIndex(graph)
        index.build(FakeEmbedder(), show_progress=False)
        metadata = json.loads(index.metadata_path.read_text(encoding="utf-8"))
        metadata["generation_id"] = "wrong-generation"
        index.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "different generations"):
            EmbeddingIndex(graph).load()

        index.build(FakeEmbedder(), force=True, show_progress=False)
        with np.load(index.vectors_path, allow_pickle=False) as data:
            node_ids = data["node_ids"].copy()
            vectors = data["vectors"].copy()
            generation_id = data["generation_id"].copy()
        vectors[0, 0] = np.nan
        np.savez_compressed(
            index.vectors_path,
            node_ids=node_ids,
            vectors=vectors,
            generation_id=generation_id,
        )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            EmbeddingIndex(graph).load()

        index.build(FakeEmbedder(), force=True, show_progress=False)
        metadata = json.loads(index.metadata_path.read_text(encoding="utf-8"))
        metadata.pop("content_hashes")
        index.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "content hashes"):
            EmbeddingIndex(graph).load()

        index.build(FakeEmbedder(), force=True, show_progress=False)
        with np.load(index.vectors_path, allow_pickle=False) as data:
            node_ids = data["node_ids"].copy()
            vectors = data["vectors"].copy() * 2.0
            generation_id = data["generation_id"].copy()
        np.savez_compressed(
            index.vectors_path,
            node_ids=node_ids,
            vectors=vectors,
            generation_id=generation_id,
        )
        with self.assertRaisesRegex(ValueError, "non-normalized"):
            EmbeddingIndex(graph).load()

    def test_invalid_search_and_link_parameters_are_rejected(self):
        graph = GraphifyGraph(self.graph_path)
        index = EmbeddingIndex(graph)
        embedder = FakeEmbedder()
        index.build(embedder, show_progress=False)
        query = embedder.encode(["embedding cache"])[0]
        with self.assertRaisesRegex(ValueError, "top_k"):
            index.search("query", query, top_k=0)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            index.search("query", np.asarray([np.nan, 0, 0, 0], dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "threshold"):
            index.similarity_pairs(threshold=float("nan"))
        strict_json = self.root / "strict.json"
        with self.assertRaisesRegex(ValueError, "Out of range float values"):
            EmbeddingIndex._write_json_atomic(strict_json, {"score": float("nan")})
        self.assertFalse(strict_json.exists())

    def test_output_cannot_bypass_in_place_backup(self):
        graph = GraphifyGraph(self.graph_path)
        index = EmbeddingIndex(graph)
        index.build(FakeEmbedder(), show_progress=False)
        original = self.graph_path.read_bytes()
        with self.assertRaisesRegex(ValueError, "cannot overwrite"):
            index.write_linked_graph([], output=self.graph_path)
        self.assertEqual(self.graph_path.read_bytes(), original)
        self.assertFalse(self.graph_path.with_suffix(".json.bak").exists())

    def test_directed_graph_gets_symmetric_semantic_edges(self):
        payload = json.loads(self.graph_path.read_text(encoding="utf-8"))
        payload["directed"] = True
        self.graph_path.write_text(json.dumps(payload), encoding="utf-8")
        graph = GraphifyGraph(self.graph_path)
        links = graph.semantic_links(
            [("cache-loader", "gpu-loader", 0.9)], model="fake"
        )
        self.assertEqual(len(links), 2)
        self.assertEqual(
            {(link["source"], link["target"]) for link in links},
            {("cache-loader", "gpu-loader"), ("gpu-loader", "cache-loader")},
        )

    def test_embedder_cleanup_when_incremental_build_fails(self):
        graph = GraphifyGraph(self.graph_path)
        embedder = ClosingEmbedder()
        args = argparse.Namespace(
            no_source_context=False,
            force_index=True,
            force=False,
            embedding_model=embedder.model_name,
            instruction=embedder.instruction,
            dtype="fp32",
        )
        with (
            patch("graphify_embeddings.cli._embedder_from_args", return_value=embedder),
            patch.object(EmbeddingIndex, "build", side_effect=RuntimeError("boom")),
            self.assertRaisesRegex(RuntimeError, "boom"),
        ):
            cli._ensure_index(graph, args, keep_embedder=True)
        self.assertTrue(embedder.closed)

    def test_official_vl_wrappers_reject_cpu_instead_of_ignoring_device(self):
        embedding_model = self.root / "embedding-model" / "scripts"
        embedding_model.mkdir(parents=True)
        (embedding_model / "qwen3_vl_embedding.py").write_text(
            "# fixture", encoding="utf-8"
        )
        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            QwenEmbedder(str(embedding_model.parent), device="cpu")

        reranker_model = self.root / "reranker-model" / "scripts"
        reranker_model.mkdir(parents=True)
        (reranker_model / "qwen3_vl_reranker.py").write_text(
            "# fixture", encoding="utf-8"
        )
        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            QwenReranker(str(reranker_model.parent), device="cpu")

    def test_default_vl_models_fail_closed_when_wrapper_is_missing(self):
        empty_snapshot = self.root / "empty-snapshot"
        empty_snapshot.mkdir()
        with (
            patch.object(
                QwenEmbedder,
                "_resolve_model_path",
                return_value=empty_snapshot,
            ),
            self.assertRaisesRegex(RuntimeError, "embedding wrapper is missing"),
        ):
            QwenEmbedder(DEFAULT_EMBEDDING_MODEL, device="cuda:0")
        with (
            patch.object(
                QwenReranker,
                "_resolve_model_path",
                return_value=empty_snapshot,
            ),
            self.assertRaisesRegex(RuntimeError, "reranker wrapper is missing"),
        ):
            QwenReranker(DEFAULT_RERANKER_MODEL, device="cuda:0")

    def test_reranker_template_compatibility_view_does_not_mutate_snapshot(self):
        model = self.root / "model"
        (model / "additional_chat_templates").mkdir(parents=True)
        (model / "chat_template.json").write_text("{}", encoding="utf-8")
        (model / "chat_template.jinja").write_text("modern", encoding="utf-8")
        (model / "config.json").write_text("{}", encoding="utf-8")
        reranker = QwenReranker.__new__(QwenReranker)
        view = reranker._processor_compatible_view(model)
        try:
            self.assertTrue((model / "chat_template.json").is_file())
            self.assertFalse((view / "chat_template.json").exists())
            self.assertTrue((view / "chat_template.jinja").is_file())
            self.assertTrue((view / "additional_chat_templates").is_dir())
            self.assertTrue((view / "config.json").is_file())
        finally:
            reranker._model_view.cleanup()

    def test_link_writes_compatible_graph_and_preserves_original(self):
        graph = GraphifyGraph(self.graph_path)
        index = EmbeddingIndex(graph)
        embedder = FakeEmbedder()
        index.build(embedder, show_progress=False)
        pairs = index.similarity_pairs(threshold=-1.0, max_neighbors=1, block_size=2)
        target, count = index.write_linked_graph(pairs)
        self.assertGreater(count, 0)
        self.assertNotEqual(target, self.graph_path)
        original = json.loads(self.graph_path.read_text(encoding="utf-8"))
        enriched = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(len(original["links"]), 1)
        semantic = [
            link
            for link in enriched["links"]
            if link.get("relation") == "semantically_similar_to"
        ]
        self.assertEqual(len(semantic), count)
        self.assertEqual(semantic[0]["confidence"], "INFERRED")
        self.assertIn("confidence_score", semantic[0])


if __name__ == "__main__":
    unittest.main()
