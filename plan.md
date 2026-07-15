# Graphify Embeddings — implemented design

Status: implemented and GPU-verified.

## Goal

Complement Graphify's exact AST/BFS/DFS graph with local semantic retrieval. Keep structural and inferred facts distinct:

- Graphify remains the source for calls/imports/paths.
- Qwen embeddings provide semantic recall.
- Qwen reranking is optional and only refines a bounded Top-K set.
- Similarity edges are marked `INFERRED`, never `EXTRACTED`.

## Pipeline

```text
graphify-out/graph.json
        |
        v
graphify-embeddings index
  - node label/docstring/metadata
  - source context around source_location
  - Qwen3-VL-Embedding-8B on cuda:0 via the official Qwen3VLEmbedder.process() wrapper used by the Comfy custom node
  - L2-normalized float32 vectors
  - node-ID + SHA256 content cache
        |
        +--> graphify-embeddings search
        |      cosine + lexical + degree
        |      Graphify neighbor expansion
        |      optional Qwen3-VL-Reranker-8B
        |
        +--> graphify-embeddings link
               graph.semantic.json by default
               graph.json only with --in-place + backup
```

## Cache

```text
graphify-out/cache/embeddings.json
  schema version, generation ID, immutable model revision, verified wrapper SHA-256,
  backend/instruction/dtype identity, dimension, node/content hashes, source-context mode

graphify-out/cache/embeddings.npz
  matching generation ID + node IDs + normalized float32 matrix
```

Unchanged vectors are reused only when the complete embedding identity and content hashes match. Changed/new node texts are embedded again. Removed nodes disappear from the rewritten matrix. Mixed-generation, non-finite, duplicate-ID, row-count, and dimension mismatches are rejected.

## GPU lifecycle

1. `auto` resolves to the first NVIDIA device (`cuda:0`).
2. The embedding model produces the index/query vector.
3. The embedding model is deleted, Python GC runs, and `torch.cuda.empty_cache()` is called.
4. Only then is the optional 8B reranker loaded.
5. Reranker cleanup repeats the same lifecycle.

The official Qwen VL reranker snapshot is loaded through its bundled `Qwen3VLReranker.process()` implementation. A temporary symlink view hides the snapshot's legacy `chat_template.json` when Transformers 5 sees both old and new template layouts; the shared Hugging Face cache is never modified.

## Verified artifact

Reference corpus:

`ComfyUI-ImageSelector-LLM/graphify-out/graph.json`

Real execution:

- 419 Graphify nodes
- Qwen3-VL-Embedding-8B
- 4096 dimensions
- `cuda:0` / RTX 5090
- incremental cache written successfully
- German semantic query returned the exact loader/cache/unloader symbols
- sequential Qwen3-VL-Reranker-8B completed on the candidate pool
- safe semantic export produced a parseable Graphify graph with zero dangling edges

## Future work

- Optional FAISS/HNSW backend for very large graphs.
- MCP wrapper exposing `index`, `search`, and `link` as tools.
- Benchmark text-only Qwen3-Embedding-8B against the current VL model on code retrieval.
- Upstream integration if Graphify issues #1/#7 settle on a compatible cache/schema.
