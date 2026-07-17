# graphify-embeddings

Local semantic search and reranking for [Graphify](https://github.com/Graphify-Labs/graphify) graphs.

Graphify provides exact AST relationships and BFS/DFS traversal. This companion adds:

1. embeddings through the official local `Qwen3VLEmbedder.process()` wrapper;
2. safe incremental node-vector caching;
3. cosine retrieval plus real Graphify-neighbor expansion;
4. optional reranking through `Qwen3VLReranker.process()`;
5. optional Graphify-compatible `semantically_similar_to` edges;
6. an authenticated local model worker with idle leases and VRAM-pressure eviction.

Everything runs locally through Python, PyTorch, Transformers and the pinned Qwen wrappers. There is no llama.cpp, Ollama, HTTP API or model server.

## Install

```bash
git clone https://github.com/THEman6989/graphify-embeddings.git
cd graphify-embeddings
./install.sh
```

The installer:

- creates a dedicated Python 3.11 `.venv`;
- synchronizes the locked CUDA/Transformers dependencies as a non-editable install;
- uses a compatible binary `flash-attn` wheel when one exists, otherwise SDPA;
- verifies CUDA from the new environment;
- atomically links `~/.local/bin/graphify-embeddings`;
- creates the default config when absent.

It does not use a ComfyUI virtual environment.

The first model operation downloads the pinned Hugging Face snapshots when they are not already cached. Pass `--local-files-only --no-worker` to an individual index/search command when network access must be forbidden.

```bash
./install.sh --no-flash-attn       # use PyTorch SDPA
./install.sh --require-flash-attn  # explicitly source-build for the detected GPUs
```

## One-command Graphify pipeline

```bash
graphify-embeddings pipeline /path/to/repository
```

The stages are deliberately sequential:

1. run `graphify extract`;
2. retry with `--code-only` if normal extraction fails;
3. require the completed `graphify-out/graph.json`;
4. incrementally index the graph;
5. write `graphify-out/graph.semantic.json` by default;
6. keep the configured models warm for the lease period.

Embedding never starts before Graphify has produced the graph.

Manual operation remains available:

```bash
graphify extract /path/to/repository --code-only
graphify-embeddings index \
  --graph /path/to/repository/graphify-out/graph.json \
  --batch-size 2 --checkpoint-size 64
graphify-embeddings link --graph /path/to/repository/graphify-out/graph.json
```

Indexing reports completed nodes, percentage, rate, and ETA after each embedding batch. On CUDA OOM, the official Qwen wrapper clears the CUDA allocator cache, halves the active batch size down to one, and retries the same work. Completed vectors are persisted in atomic checkpoint shards (64 nodes by default), so an interrupted compatible run resumes instead of restarting. `--batch-size` controls simultaneous GPU work; `--checkpoint-size` independently controls how often resumable CPU-side shards are published.

## Search and reranking

```bash
graphify-embeddings search \
  "where is the embedding cache invalidated?" \
  --graph /path/to/repository/graphify-out/graph.json \
  --top-k 10 --neighbors 1
```

Reranking is enabled by default in `config.toml`. It can be overridden per request:

```bash
graphify-embeddings search "trace VRAM cleanup" --rerank
graphify-embeddings search "fast embedding-only search" --no-rerank
graphify-embeddings search "machine output" --json
```

The expensive cross-attention reranker sees only the configured candidate pool.

## Persistent worker and 60-second lease

```bash
graphify-embeddings worker start
graphify-embeddings worker status
graphify-embeddings worker ping --models embedding reranker
graphify-embeddings worker warm --models embedding reranker
graphify-embeddings worker unload --models reranker
graphify-embeddings worker stop
```

The worker uses a same-user Unix socket and random `0600` token below `$XDG_RUNTIME_DIR/graphify-embeddings/`. It opens no TCP port and serializes model operations.

- A model is unloaded after 60 idle seconds by default.
- The last validated graph/index pair stays in worker RAM for the same idle lease, avoiding repeated JSON/NPZ parsing and decompression.
- Graph, metadata or NPZ replacement invalidates the RAM index synchronously; generation IDs and before/after file signatures fail closed.
- `ping` extends the lease only for models that are already resident.
- `warm` loads missing models and starts a new lease.
- `auto_start = false` reuses an already running worker but falls back to one-shot execution instead of spawning one.
- Runtime-affecting config changes require `graphify-embeddings worker stop` first; commands never reuse a worker whose effective config fingerprint differs from `--config`.
- `--no-worker` uses the one-shot path and closes everything immediately.
- Stop, SIGTERM, SIGINT and fatal exits close both models and remove runtime files.
- When the last model and RAM index are evicted, the worker exits to release process RAM and CUDA context VRAM too. The next compatible CLI job auto-starts it unless `auto_start` is disabled.

## Multi-GPU and higher-priority applications

Measured BF16 footprints on the tested host:

```text
Qwen3-VL-Embedding-8B:  about 15.21 GiB
Qwen3-VL-Reranker-8B:   about 16.54 GiB peak reservation
combined:                about 30.43 GiB plus runtime overhead
RTX 5090 usable total:   about 31.40 GiB
```

Both models are therefore not safe co-residents on the 5090. Default split placement is:

```text
Reranker  -> cuda:0 / RTX 5090
Embedder  -> cuda:1 / RTX 3090
```

NVML is checked before loads and between requests. The worker evicts models when free VRAM falls below the configured reserve or another process on that GPU crosses the high-priority memory threshold. If possible, the requested model is reloaded on the other GPU; otherwise the request fails safely instead of forcing an OOM. ComfyUI and games therefore override an idle lease or ping.

An already running CUDA kernel cannot be safely interrupted mid-inference. Pressure protection applies before requests and between requests, not by killing active kernels.

The official VL wrappers are CUDA-only. `--device cpu` is rejected instead of silently selecting a GPU.

## Configuration

```bash
graphify-embeddings config path
graphify-embeddings config init
graphify-embeddings config show
```

See `config.example.toml`. Important options:

```toml
[worker]
enabled = true
auto_start = true
idle_timeout_seconds = 60
reranker_enabled = true

[gpu]
placement_policy = "split"
preferred_gpu = 0
secondary_gpu = 1
min_free_gib = 6
high_priority_process_mib = 2048

[attention]
backend = "auto" # flash_attention_2 when importable, otherwise sdpa
```

CLI `--rerank` and `--no-rerank` override the configured reranker default for one search.

## Cache files

```text
graphify-out/cache/embeddings.json  identity, hashes, dimensions, generation
graphify-out/cache/embeddings.npz   normalized float32 vectors and node IDs
graphify-out/cache/.embeddings.lock serialized writer lock
graphify-out/cache/embedding-checkpoint/ temporary resumable shard generation
graphify-out/graph.semantic.json     optional semantic Graphify graph
```

The cache is JSON+NPZ rather than SQLite because retrieval uses one contiguous dense matrix. Cache schema 6 includes model/revision, exact wrapper SHA-256, local artifact fingerprint, instruction, dtype, attention backend and document schema. Any mismatch invalidates reuse.

Publication is atomic and generation-paired. Non-finite or non-normalized vectors, duplicate IDs, graph-membership mismatch, dimensions, incomplete hashes and mixed generations fail closed.

Incremental reuse is keyed by node ID plus content hash and remains valid when nodes are added or removed. Checkpoint shards additionally bind every row to the complete embedding identity and current content hash. They are deleted only after the final cache generation has been published successfully.

## Model trust boundary

Defaults:

- `Qwen/Qwen3-VL-Embedding-8B`
- `Qwen/Qwen3-VL-Reranker-8B`

The exact official wrapper bytes must match pinned SHA-256 values before those same bytes are compiled and executed. Alternative remote models require explicit immutable 40-character revisions. Local model directories receive an artifact fingerprint.

```bash
graphify-embeddings index \
  --embedding-model Qwen/Qwen3-Embedding-8B \
  --embedding-revision <40-character-commit> \
  --no-worker
```

## Why both Graphify and embeddings?

- Graphify answers exact structural questions: calls, imports, ownership and paths.
- Embeddings recover naming variants, cross-language concepts and semantic similarity.
- `--neighbors` attaches real Graphify relations instead of pretending cosine similarity is a call edge.
- Reranking applies expensive cross-attention only to a small candidate set.

## Development

```bash
.venv/bin/python -m unittest discover -s tests -v
uvx ruff check src tests
uvx ruff format --check src tests
```

Apache-2.0 licensed.
