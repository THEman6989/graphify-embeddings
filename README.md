# graphify-embeddings

Local semantic search and reranking for [Graphify](https://github.com/Graphify-Labs/graphify) graphs.

Graphify gives exact AST relationships and BFS/DFS traversal. This tool adds a complementary local retrieval path:

1. embed Graphify nodes with Qwen3-VL-Embedding-8B through the official `Qwen3VLEmbedder.process()` wrapper used by the Comfy custom node;
2. cache vectors incrementally by node ID + content hash;
3. retrieve semantic candidates with cosine similarity;
4. expand each hit through real Graphify edges;
5. optionally rerank Top-K with Qwen3-VL-Reranker-8B;
6. optionally write `semantically_similar_to` edges into a Graphify-compatible graph.

Everything runs locally and directly through the model's Python wrapper. No API key, llama.cpp, Ollama, or model server is needed.

## Hardware behavior

`--device auto` deliberately chooses the first NVIDIA GPU (`cuda:0`). On Amin's machine that is the RTX 5090. The embedding model is unloaded before the optional 8B reranker is loaded, so both models do not occupy VRAM at the same time.

The Comfy-compatible official VL wrappers require CUDA; `--device cpu` is rejected instead of silently using a GPU. CPU mode remains available only for explicitly selected alternative SentenceTransformers-compatible models.

## Install

```bash
cd ~/experi/krams/graphify-embeddings
uv venv
uv pip install -e '.[gpu]'
```

If the ComfyUI environment already contains Torch and Sentence Transformers:

```bash
/home/amin/experi/sabilitymatrix/Data/Packages/ComfyUI_arc7/venv/bin/python -m pip install -e . --no-deps
```

## Quick start

Build a Graphify graph first:

```bash
graphify extract /path/to/repo --code-only
```

Index it:

```bash
graphify-embeddings index \
  --graph /path/to/repo/graphify-out/graph.json
```

Semantic + structural search:

```bash
graphify-embeddings search \
  "where is the embedding cache invalidated?" \
  --graph /path/to/repo/graphify-out/graph.json \
  --top-k 10 \
  --neighbors 1
```

Add Qwen reranking:

```bash
graphify-embeddings search \
  "trace model loading and VRAM cleanup" \
  --graph /path/to/repo/graphify-out/graph.json \
  --rerank \
  --candidate-k 24 \
  --top-k 8
```

Machine-readable output:

```bash
graphify-embeddings search "semantic query" --graph graphify-out/graph.json --json
```

## Add semantic edges

Safe default: write a second graph without touching Graphify's original:

```bash
graphify-embeddings link --graph graphify-out/graph.json --threshold 0.82
# writes graphify-out/graph.semantic.json
```

Explicit in-place update:

```bash
graphify-embeddings link --graph graphify-out/graph.json --threshold 0.82 --in-place
# creates graphify-out/graph.json.bak first
```

Edges use Graphify's real `links` schema:

```json
{
  "source": "node-a",
  "target": "node-b",
  "relation": "semantically_similar_to",
  "type": "semantically_similar_to",
  "confidence": "INFERRED",
  "confidence_score": 0.87,
  "weight": 0.87
}
```

## Cache files

Beside the graph:

```text
graphify-out/cache/embeddings.json   model/revision/wrapper/prompt identity, dimensions, content hashes, generation ID
graphify-out/cache/embeddings.npz    normalized float32 vectors, node IDs, matching generation ID
```

Changing the instruction, dtype, backend, pinned model revision, local model artifact, local wrapper script, or document-construction schema invalidates the relevant cache identity. Local VL wrapper scripts must match the audited official SHA-256; mutable custom Python is never executed. Alternative remote models require an explicit immutable 40-character commit revision. A writer lock serializes concurrent builds; a generation ID detects partial/crash-mixed metadata/vector writes. Non-finite, non-normalized, duplicate-ID, graph-membership, content-hash, row-count, and dimension mismatches are rejected. Only new or changed node texts are embedded on a compatible subsequent `index` run. Source context around `source_location` is included when the referenced source file exists.

## Models

Defaults:

- `Qwen/Qwen3-VL-Embedding-8B`
- `Qwen/Qwen3-VL-Reranker-8B`

The text-only equivalents can be selected explicitly:

```bash
graphify-embeddings index \
  --embedding-model Qwen/Qwen3-Embedding-8B \
  --embedding-revision <40-character-commit>
graphify-embeddings search "..." --rerank \
  --reranker-model Qwen/Qwen3-Reranker-8B \
  --reranker-revision <40-character-commit>
```

## Why both Graphify and embeddings?

- Graphify edges answer exact structural questions: calls, imports, ownership, paths.
- Qwen embeddings recover naming variants, cross-language concepts and semantically similar code.
- The reranker spends expensive cross-attention only on a small candidate set.
- `--neighbors` attaches actual Graphify relationships to semantic hits rather than pretending cosine similarity is a call edge.

## Development

```bash
python -m unittest discover -s tests -v
```

Apache-2.0 licensed.
