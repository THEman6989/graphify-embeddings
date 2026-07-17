from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import numpy as np

from graphify_embeddings import cli
from graphify_embeddings.graph import GraphifyGraph
from graphify_embeddings.index import EmbeddingIndex
from graphify_embeddings.models import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_REVISION,
    DEFAULT_RERANKER_MODEL,
    QwenEmbedder,
    QwenReranker,
    local_model_fingerprint,
    resolve_device,
    resolve_model_revision,
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


class BatchedFakeEmbedder(FakeEmbedder):
    batch_size = 2


class FailAfterFirstBatchEmbedder(BatchedFakeEmbedder):
    def encode(self, texts, *, show_progress=False):
        if self.calls:
            raise RuntimeError("simulated interruption")
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


class FakeTorch:
    float32 = object()
    float16 = object()
    bfloat16 = object()

    class cuda:
        selected = None

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 2

        @classmethod
        def set_device(cls, index):
            cls.selected = index

        @staticmethod
        def empty_cache():
            return None


class OOMAboveTwoModel:
    def __init__(self):
        self.batch_sizes: list[int] = []

    def process(self, items, normalize=True):
        self.batch_sizes.append(len(items))
        if len(items) > 2:
            raise RuntimeError("CUDA out of memory")
        return np.asarray(
            [[float(index + 1), 1.0, 0.5, 0.25] for index, _ in enumerate(items)],
            dtype=np.float32,
        )


class OOMAboveOneOrderedModel:
    def __init__(self):
        self.batch_sizes: list[int] = []

    def process(self, items, normalize=True):
        self.batch_sizes.append(len(items))
        if len(items) > 1:
            raise RuntimeError("CUDA out of memory")
        value = float(items[0]["text"])
        return np.asarray([[value, 1.0, 0.5, 0.25]], dtype=np.float32)


class AlwaysOOMModel:
    def process(self, items, normalize=True):
        raise RuntimeError("CUDA out of memory")


class NonOOMFailureModel:
    def process(self, items, normalize=True):
        raise RuntimeError("broken model contract")


class GenericAllocationOOMModel:
    def process(self, items, normalize=True):
        raise RuntimeError("memory allocation failed with oom")


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

    @staticmethod
    def _official_embedder_for_test(model, *, batch_size=4):
        embedder = object.__new__(QwenEmbedder)
        embedder.backend = "official_vl_wrapper"
        embedder.model = model
        embedder.instruction = "test"
        embedder.batch_size = batch_size
        embedder.__dict__["torch"] = FakeTorch
        embedder.device = "cuda:1"
        return embedder

    def test_index_parser_accepts_checkpoint_size(self):
        args = cli.build_parser().parse_args(["index", "--checkpoint-size", "128"])
        self.assertEqual(args.checkpoint_size, 128)

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

    def test_incremental_cache_reuses_vectors_when_node_set_changes(self):
        graph = GraphifyGraph(self.graph_path)
        EmbeddingIndex(graph).build(FakeEmbedder(), show_progress=False)

        updated = json.loads(self.graph_path.read_text(encoding="utf-8"))
        updated["nodes"] = updated["nodes"][:2]
        updated["nodes"].append(
            {
                "id": "new-node",
                "label": "new_node",
                "docstring": "A newly added node",
                "source_file": "new.py",
                "source_location": "L1",
                "community": 2,
            }
        )
        updated["links"] = []
        self.graph_path.write_text(json.dumps(updated), encoding="utf-8")

        embedder = FakeEmbedder()
        stats = EmbeddingIndex(GraphifyGraph(self.graph_path)).build(
            embedder, show_progress=False
        )

        self.assertEqual(stats["embedded"], 1)
        self.assertEqual(stats["reused"], 2)
        self.assertEqual(len(embedder.calls), 1)
        self.assertEqual(len(embedder.calls[0]), 1)

    def test_build_reports_progress_after_each_embedding_batch(self):
        graph = GraphifyGraph(self.graph_path)
        index = EmbeddingIndex(graph)
        stderr = io.StringIO()

        with patch("sys.stderr", stderr):
            index.build(BatchedFakeEmbedder(), show_progress=True)

        output = stderr.getvalue()
        self.assertIn("Embedding progress: 2/3 (66.7%)", output)
        self.assertIn("Embedding progress: 3/3 (100.0%)", output)
        self.assertIn("ETA", output)

    def test_qwen_embedder_retries_cuda_oom_with_smaller_batches(self):
        embedder = self._official_embedder_for_test(OOMAboveTwoModel())
        stderr = io.StringIO()

        with (
            patch("graphify_embeddings.models._empty_device_cache") as empty_cache,
            patch("sys.stderr", stderr),
        ):
            vectors = embedder.encode(["a", "b", "c", "d"])

        self.assertEqual(vectors.shape, (4, 4))
        self.assertEqual(embedder.model.batch_sizes, [4, 2, 2])
        self.assertEqual(embedder.batch_size, 2)
        empty_cache.assert_called_once_with(FakeTorch, "cuda:1")
        self.assertIn("CUDA OOM", stderr.getvalue())
        self.assertIn("batch size 4 -> 2", stderr.getvalue())

    def test_qwen_embedder_reduces_oom_to_one_and_preserves_order(self):
        model = OOMAboveOneOrderedModel()
        embedder = self._official_embedder_for_test(model)

        with (
            patch("graphify_embeddings.models._empty_device_cache"),
            patch("sys.stderr", io.StringIO()),
        ):
            vectors = embedder.encode(["1", "2", "3"])

        expected = np.asarray(
            [[1.0, 1.0, 0.5, 0.25], [2.0, 1.0, 0.5, 0.25], [3.0, 1.0, 0.5, 0.25]],
            dtype=np.float32,
        )
        expected /= np.linalg.norm(expected, axis=1, keepdims=True)
        np.testing.assert_allclose(vectors, expected)
        self.assertEqual(model.batch_sizes, [3, 2, 1, 1, 1])
        self.assertEqual(embedder.batch_size, 1)

    def test_qwen_embedder_reports_single_document_oom(self):
        embedder = self._official_embedder_for_test(AlwaysOOMModel(), batch_size=1)

        with (
            patch("graphify_embeddings.models._empty_device_cache") as empty_cache,
            self.assertRaisesRegex(RuntimeError, "single document"),
        ):
            embedder.encode(["1"])

        empty_cache.assert_called_once_with(FakeTorch, "cuda:1")

    def test_qwen_embedder_does_not_swallow_non_oom_runtime_error(self):
        embedder = self._official_embedder_for_test(NonOOMFailureModel())

        with (
            patch("graphify_embeddings.models._empty_device_cache") as empty_cache,
            self.assertRaisesRegex(RuntimeError, "broken model contract"),
        ):
            embedder.encode(["1"])

        empty_cache.assert_not_called()

    def test_qwen_embedder_does_not_misclassify_generic_allocation_oom(self):
        embedder = self._official_embedder_for_test(GenericAllocationOOMModel())

        with (
            patch("graphify_embeddings.models._empty_device_cache") as empty_cache,
            self.assertRaisesRegex(RuntimeError, "memory allocation failed with oom"),
        ):
            embedder.encode(["1"])

        empty_cache.assert_not_called()

    def test_qwen_embedder_releases_active_exception_before_oom_cleanup(self):
        embedder = self._official_embedder_for_test(OOMAboveTwoModel())
        cleanup_exception_states = []

        def record_exception_state(_torch, _device):
            cleanup_exception_states.append(sys.exc_info())

        with (
            patch(
                "graphify_embeddings.models._empty_device_cache",
                side_effect=record_exception_state,
            ),
            patch("sys.stderr", io.StringIO()),
        ):
            embedder.encode(["a", "b", "c", "d"])

        self.assertEqual(len(cleanup_exception_states), 1)
        self.assertEqual(cleanup_exception_states[0], (None, None, None))

    def test_build_rejects_invalid_checkpoint_size_without_writing_cache(self):
        graph = GraphifyGraph(self.graph_path)

        for invalid in (True, "64", 1.5, 0, -1):
            with self.subTest(invalid=invalid):
                index = EmbeddingIndex(graph)
                with self.assertRaisesRegex(
                    ValueError, "checkpoint_size must be a positive integer"
                ):
                    index.build(
                        BatchedFakeEmbedder(),
                        show_progress=False,
                        checkpoint_size=cast(Any, invalid),
                    )
                self.assertFalse(index.cache_dir.exists())

    def test_build_resumes_from_persistent_checkpoint_shards(self):
        graph = GraphifyGraph(self.graph_path)
        interrupted = EmbeddingIndex(graph)

        with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
            interrupted.build(
                FailAfterFirstBatchEmbedder(),
                show_progress=False,
                checkpoint_size=2,
            )

        self.assertTrue(interrupted.checkpoint_manifest_path.is_file())
        self.assertEqual(len(list(interrupted.checkpoint_dir.glob("shard-*.npz"))), 1)

        resumed = EmbeddingIndex(graph)
        embedder = BatchedFakeEmbedder()
        stats = resumed.build(
            embedder,
            show_progress=False,
            checkpoint_size=2,
        )

        self.assertEqual(stats["embedded"], 3)
        self.assertEqual(stats["computed"], 1)
        self.assertEqual(stats["resumed"], 2)
        self.assertEqual(len(embedder.calls), 1)
        self.assertEqual(len(embedder.calls[0]), 1)
        self.assertFalse(resumed.checkpoint_dir.exists())

    def test_fully_resumed_build_reports_completion_progress(self):
        graph = GraphifyGraph(self.graph_path)
        interrupted = EmbeddingIndex(graph)

        with (
            patch.object(
                EmbeddingIndex,
                "_write_vectors_atomic",
                side_effect=RuntimeError("simulated final publication interruption"),
            ),
            self.assertRaisesRegex(RuntimeError, "final publication interruption"),
        ):
            interrupted.build(
                BatchedFakeEmbedder(),
                show_progress=False,
                checkpoint_size=2,
            )

        stderr = io.StringIO()
        resumed = EmbeddingIndex(graph)
        with patch("sys.stderr", stderr):
            stats = resumed.build(
                BatchedFakeEmbedder(),
                show_progress=True,
                checkpoint_size=2,
            )

        self.assertEqual(stats["computed"], 0)
        self.assertEqual(stats["resumed"], 3)
        output = stderr.getvalue()
        self.assertEqual(output.count("Embedding progress:"), 1)
        self.assertIn("Embedding progress: 3/3 (100.0%)", output)
        self.assertFalse(resumed.checkpoint_dir.exists())

    def test_force_build_discards_compatible_checkpoint_shards(self):
        graph = GraphifyGraph(self.graph_path)
        interrupted = EmbeddingIndex(graph)

        with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
            interrupted.build(
                FailAfterFirstBatchEmbedder(),
                show_progress=False,
                checkpoint_size=2,
            )

        forced = EmbeddingIndex(graph)
        embedder = BatchedFakeEmbedder()
        stats = forced.build(
            embedder,
            force=True,
            show_progress=False,
            checkpoint_size=2,
        )

        self.assertEqual(stats["embedded"], 3)
        self.assertEqual(stats["computed"], 3)
        self.assertEqual(stats["resumed"], 0)
        self.assertEqual([len(call) for call in embedder.calls], [2, 1])
        self.assertFalse(forced.checkpoint_dir.exists())

    def test_build_discards_non_normalized_checkpoint_shard(self):
        graph = GraphifyGraph(self.graph_path)
        interrupted = EmbeddingIndex(graph)

        with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
            interrupted.build(
                FailAfterFirstBatchEmbedder(),
                show_progress=False,
                checkpoint_size=2,
            )

        shard_path = next(interrupted.checkpoint_dir.glob("shard-*.npz"))
        with np.load(shard_path, allow_pickle=False) as data:
            node_ids = data["node_ids"].copy()
            content_hashes = data["content_hashes"].copy()
            vectors = data["vectors"].copy()
            generation_id = data["generation_id"].copy()
        np.savez_compressed(
            shard_path,
            node_ids=node_ids,
            content_hashes=content_hashes,
            vectors=vectors * 2.0,
            generation_id=generation_id,
        )

        resumed = EmbeddingIndex(graph)
        embedder = BatchedFakeEmbedder()
        stats = resumed.build(
            embedder,
            show_progress=False,
            checkpoint_size=2,
        )

        self.assertEqual(stats["computed"], 3)
        self.assertEqual(stats["resumed"], 0)
        self.assertEqual([len(call) for call in embedder.calls], [2, 1])
        self.assertFalse(resumed.checkpoint_dir.exists())

    def test_build_discards_truncated_checkpoint_shard(self):
        graph = GraphifyGraph(self.graph_path)
        interrupted = EmbeddingIndex(graph)

        with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
            interrupted.build(
                FailAfterFirstBatchEmbedder(),
                show_progress=False,
                checkpoint_size=2,
            )

        shard_path = next(interrupted.checkpoint_dir.glob("shard-*.npz"))
        shard_path.write_bytes(b"truncated-npz")

        resumed = EmbeddingIndex(graph)
        embedder = BatchedFakeEmbedder()
        stats = resumed.build(
            embedder,
            show_progress=False,
            checkpoint_size=2,
        )

        self.assertEqual(stats["computed"], 3)
        self.assertEqual(stats["resumed"], 0)
        self.assertFalse(resumed.checkpoint_dir.exists())

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
        with (
            patch(
                "graphify_embeddings.models._imports",
                return_value=(FakeTorch, None, None),
            ),
            self.assertRaisesRegex(RuntimeError, "requires CUDA"),
        ):
            QwenEmbedder(str(embedding_model.parent), device="cpu")

        reranker_model = self.root / "reranker-model" / "scripts"
        reranker_model.mkdir(parents=True)
        (reranker_model / "qwen3_vl_reranker.py").write_text(
            "# fixture", encoding="utf-8"
        )
        with (
            patch(
                "graphify_embeddings.models._imports",
                return_value=(FakeTorch, None, None),
            ),
            self.assertRaisesRegex(RuntimeError, "requires CUDA"),
        ):
            QwenReranker(str(reranker_model.parent), device="cpu")

    def test_default_vl_models_fail_closed_when_wrapper_is_missing(self):
        empty_snapshot = self.root / "empty-snapshot"
        empty_snapshot.mkdir()
        with (
            patch(
                "graphify_embeddings.models._imports",
                return_value=(FakeTorch, None, None),
            ),
            patch.object(
                QwenEmbedder,
                "_resolve_model_path",
                return_value=empty_snapshot,
            ),
            self.assertRaisesRegex(RuntimeError, "embedding wrapper is missing"),
        ):
            QwenEmbedder(DEFAULT_EMBEDDING_MODEL, device="cuda:0")
        with (
            patch(
                "graphify_embeddings.models._imports",
                return_value=(FakeTorch, None, None),
            ),
            patch.object(
                QwenReranker,
                "_resolve_model_path",
                return_value=empty_snapshot,
            ),
            self.assertRaisesRegex(RuntimeError, "reranker wrapper is missing"),
        ):
            QwenReranker(DEFAULT_RERANKER_MODEL, device="cuda:0")

    def test_local_model_fingerprint_and_wrapper_trust_boundary(self):
        model = self.root / "mutable-model"
        scripts = model / "scripts"
        scripts.mkdir(parents=True)
        config = model / "config.json"
        config.write_text('{"version": 1}', encoding="utf-8")
        first = local_model_fingerprint(model)
        config.write_text('{"version": 2}', encoding="utf-8")
        second = local_model_fingerprint(model)
        self.assertNotEqual(first, second)

        (scripts / "qwen3_vl_embedding.py").write_text(
            "# untrusted local code", encoding="utf-8"
        )
        with (
            patch(
                "graphify_embeddings.models._imports",
                return_value=(FakeTorch, None, None),
            ),
            self.assertRaisesRegex(RuntimeError, "official SHA-256"),
        ):
            QwenEmbedder(str(model), device="cuda:0")

        (scripts / "qwen3_vl_embedding.py").unlink()
        (scripts / "qwen3_vl_reranker.py").write_text(
            "# untrusted local code", encoding="utf-8"
        )
        with (
            patch(
                "graphify_embeddings.models._imports",
                return_value=(FakeTorch, None, None),
            ),
            self.assertRaisesRegex(RuntimeError, "official SHA-256"),
        ):
            QwenReranker(str(model), device="cuda:0")

    def test_verified_wrapper_executes_exact_bytes_without_future_inheritance(self):
        script = self.root / "verified_wrapper.py"
        source = b"class Qwen3VLEmbedder:\n    value: int\n"
        script.write_bytes(source)
        expected = hashlib.sha256(source).hexdigest()
        try:
            with patch(
                "graphify_embeddings.models.DEFAULT_EMBEDDING_WRAPPER_SHA256",
                expected,
            ):
                wrapper, actual = QwenEmbedder._load_wrapper(script)
            self.assertEqual(actual, expected)
            self.assertIs(wrapper.__annotations__["value"], int)
        finally:
            sys.modules.pop("graphify_embeddings._qwen3_vl_embedding", None)

    def test_remote_revisions_and_cuda_device_syntax_are_strict(self):
        self.assertEqual(
            resolve_model_revision(
                DEFAULT_EMBEDDING_MODEL,
                None,
                default_model=DEFAULT_EMBEDDING_MODEL,
                default_revision=DEFAULT_EMBEDDING_REVISION,
            ),
            DEFAULT_EMBEDDING_REVISION,
        )
        commit = "a" * 40
        self.assertEqual(
            resolve_model_revision(
                "example/alternative",
                commit,
                default_model=DEFAULT_EMBEDDING_MODEL,
                default_revision=DEFAULT_EMBEDDING_REVISION,
            ),
            commit,
        )
        for revision in (None, "main", "abc123"):
            with self.assertRaisesRegex(ValueError, "immutable 40-character"):
                resolve_model_revision(
                    "example/alternative",
                    revision,
                    default_model=DEFAULT_EMBEDDING_MODEL,
                    default_revision=DEFAULT_EMBEDDING_REVISION,
                )
        with patch(
            "graphify_embeddings.models._imports",
            return_value=(FakeTorch, None, None),
        ):
            self.assertEqual(resolve_device("cuda"), "cuda:0")
            self.assertEqual(resolve_device("cuda:1"), "cuda:1")
            for invalid in ("cudafoo", "cuda:-1", "cuda:", "cuda:1x"):
                with self.assertRaisesRegex(ValueError, "Unsupported device"):
                    resolve_device(invalid)

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

    @unittest.skipUnless(shutil.which("graphify"), "Graphify CLI is not installed")
    def test_enriched_directed_graph_round_trips_through_graphify_cli(self):
        payload = json.loads(self.graph_path.read_text(encoding="utf-8"))
        payload["directed"] = True
        self.graph_path.write_text(json.dumps(payload), encoding="utf-8")
        graph = GraphifyGraph(self.graph_path)
        index = EmbeddingIndex(graph)
        index.build(FakeEmbedder(), show_progress=False)
        target, count = index.write_linked_graph([("cache-loader", "gpu-loader", 0.9)])
        self.assertEqual(count, 2)
        result = subprocess.run(
            [
                "graphify",
                "explain",
                "load_embedding_cache",
                "--graph",
                str(target),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("load_embedding_cache", result.stdout)


if __name__ == "__main__":
    unittest.main()
