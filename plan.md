# Graphify Embeddings — implemented design

Status: version 0.2 worker architecture, dedicated virtual environment, and CUDA gates implemented.

## Goal

Complement Graphify's exact AST/BFS/DFS graph with local semantic retrieval while keeping structural and inferred facts distinct.

- Graphify remains the source for calls/imports/paths.
- Qwen embeddings provide semantic recall.
- Qwen reranking is configurable and refines only a bounded candidate set.
- Similarity edges are `INFERRED`, never `EXTRACTED`.

## Pipeline

```text
graphify-embeddings pipeline <repository>
  |
  +--> graphify extract (fallback: --code-only)
  +--> require graphify-out/graph.json
  +--> worker:index
  |      official Qwen3VLEmbedder.process()
  |      incremental normalized float32 cache
  +--> graph.semantic.json
  +--> worker:warm embedding + optional reranker
```

Search combines cosine retrieval, lexical/degree signals, Graphify-neighbor expansion, and optional official `Qwen3VLReranker.process()` scoring.

## Cache

```text
graphify-out/cache/embeddings.json
  schema 6, generation ID, immutable model revision/local fingerprint,
  verified wrapper SHA-256, backend/instruction/dtype/attention identity,
  dimensions, node/content hashes and source-context mode

graphify-out/cache/embeddings.npz
  matching generation ID + node IDs + normalized float32 matrix
```

Reuse requires the complete identity, document schema, graph membership and content hashes. A writer lock serializes builds. Mixed generations, non-finite/non-normalized vectors, duplicate IDs and dimension/membership/hash mismatches fail closed.

## Worker and GPU lifecycle

A single authenticated same-user Unix-socket worker owns CUDA model objects. Operations are serialized because the official wrappers rely on process-global CUDA current-device state.

Defaults:

- idle lease: 60 seconds;
- reranking: enabled, CLI-overridable;
- pressure poll: 2 seconds;
- safe free-VRAM reserve: 6 GiB;
- high-priority external-process threshold: 2048 MiB;
- attention: FlashAttention 2 when the package imports, otherwise PyTorch SDPA.

Measured BF16 usage:

- embedder: approximately 15.21 GiB;
- reranker: approximately 16.54 GiB peak reservation;
- combined models exceed safe usable RTX 5090 capacity after runtime overhead.

Default placement is therefore reranker on RTX 5090 (`cuda:0`) and embedder on RTX 3090 (`cuda:1`). Before model loads and between requests, NVML pressure can evict either model. If possible the requested model migrates by unload/reload to the other GPU. Active CUDA kernels are never killed mid-operation.

Each model close deletes the wrapper, runs Python GC and calls `torch.cuda.empty_cache()`. Worker stop/signals/fatal exit close both models and remove socket/PID/token files. `ping` extends only existing leases; `warm` loads missing models. VRAM pressure overrides both.

## Isolation and installation

`install.sh` creates repository `.venv` with Python 3.11 and pinned Torch/Transformers/Qwen/NVML packages, optionally builds `flash-attn`, validates CUDA, and atomically points `~/.local/bin/graphify-embeddings` at the dedicated environment. No ComfyUI environment is reused.

## Trust boundary

Official wrapper files are read once, SHA-256 verified, then those exact bytes are compiled with `dont_inherit=True`. Local model artifacts are fingerprinted. Alternative remote models require immutable 40-character revisions. The temporary reranker processor view never mutates the Hugging Face cache.

## Verified reference artifact

Reference corpus: `ComfyUI-ImageSelector-LLM/graphify-out/graph.json`

Previously verified one-shot path:

- 419 nodes, 4096 dimensions;
- real embedding and reranking on CUDA;
- 468 semantic links;
- Graphify parsed and explained the semantic export.

The worker-specific real multi-GPU, lease, pressure and dedicated-venv results are recorded after the final gates.

## Future work

- Optional FAISS/HNSW for very large graphs.
- MCP wrapper exposing index/search/link/lease operations.
- Benchmark the text-only Qwen models against the VL defaults.
