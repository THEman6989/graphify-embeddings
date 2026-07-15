from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from graphify_embeddings.graph import GraphifyGraph
from graphify_embeddings.index import EmbeddingIndex
from graphify_embeddings.models import QwenReranker


class FakeEmbedder:
    model_name = "fake/qwen-8b"
    device = "cuda:0"

    def __init__(self):
        self.calls: list[list[str]] = []

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
            link for link in enriched["links"] if link.get("relation") == "semantically_similar_to"
        ]
        self.assertEqual(len(semantic), count)
        self.assertEqual(semantic[0]["confidence"], "INFERRED")
        self.assertIn("confidence_score", semantic[0])


if __name__ == "__main__":
    unittest.main()
