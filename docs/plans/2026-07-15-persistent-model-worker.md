# Persistent Model Worker Implementation Plan

> **For Hermes:** Implement task-by-task with strict RED→GREEN→REFACTOR and verify each gate before proceeding.

**Goal:** Add an optional local model worker that keeps the official Qwen3-VL embedder/reranker resident for a configurable 60-second idle lease, supports ping and reranker toggling, places the models safely across the RTX 5090/3090, evicts them under external VRAM pressure, and installs into its own reproducible uv virtual environment.

**Architecture:** A single-process Unix-domain-socket worker owns all CUDA model objects; requests are serialized because the official wrappers and CUDA current-device state are not thread-safe. CLI `index` and `search` auto-start and use the worker unless disabled. A pure placement/config layer is independently testable. The worker checks idle deadlines and NVML pressure between requests, never closes a model during active inference, and exits/cleans all models on stop or fatal error.

**Tech Stack:** Python 3.11, stdlib `socketserver`/Unix sockets, TOML via `tomllib`, PyTorch/Transformers official Qwen wrappers, `nvidia-ml-py`, uv, unittest.

---

### Task 1: Configuration contract

**Files:**
- Create: `src/graphify_embeddings/config.py`
- Create: `tests/test_worker.py`
- Create: `config.example.toml`

**Steps:**
1. Write failing tests for defaults: worker enabled, 60-second idle timeout, reranker enabled, split auto-placement, 6 GiB reserve, 2-second pressure polling, and local XDG paths.
2. Test TOML overrides and rejection of invalid timeout/device/reserve values.
3. Implement frozen dataclasses and `load_config()` using `tomllib`; config precedence is CLI override > config file > default.
4. Verify narrow tests, then full suite.

### Task 2: GPU inventory and deterministic placement

**Files:**
- Create: `src/graphify_embeddings/gpu.py`
- Test: `tests/test_worker.py`

**Steps:**
1. Write failing pure tests using fake GPU snapshots for the measured model footprints: embedder 15.25 GiB and reranker 16.5 GiB safe reservation.
2. Prove both are never placed together on the 31.4 GiB 5090 when baseline free memory is 29.85 GiB.
3. Prove default split places reranker on preferred GPU 0 (5090) and embedder on GPU 1 (3090).
4. Prove explicit device overrides work only if capacity is available.
5. Prove pressure detection triggers when free VRAM falls below reserve or an external process exceeds the configured high-use threshold.
6. Implement optional NVML inventory with a clear fail-closed error when automatic multi-GPU placement is requested but inventory is unavailable.

### Task 3: Lease-aware model manager

**Files:**
- Create: `src/graphify_embeddings/manager.py`
- Modify: `src/graphify_embeddings/models.py`
- Test: `tests/test_worker.py`

**Steps:**
1. Write failing tests with fake clock/model factories for lazy load, reuse before 60 seconds, unload at 60 seconds, and ping extending only loaded-model leases.
2. Test reranker-disabled behavior and explicit unload.
3. Test pressure eviction order: idle model first, then both if required.
4. Test model placement and device-specific factories.
5. Implement `ModelManager` with one lock, active-request guard, monotonic deadlines, `tick()`, `ping()`, `status()`, `unload()`, and `close()`.
6. Add optional attention implementation to Qwen wrapper construction; `auto` uses FlashAttention 2 only when importable, otherwise SDPA. Cache identity records the requested/effective attention backend because it can affect numerics.
7. Verify all model cleanup still runs `gc.collect()` and `torch.cuda.empty_cache()`.

### Task 4: Authenticated local Unix worker

**Files:**
- Create: `src/graphify_embeddings/worker.py`
- Create: `src/graphify_embeddings/client.py`
- Test: `tests/test_worker.py`

**Steps:**
1. Write failing integration tests using a temporary Unix socket and fake manager for `status`, `ping`, `unload`, and `stop`.
2. Test socket mode `0600`, stale socket recovery, malformed request rejection, request-size cap, and same-user token authentication.
3. Add serialized `index` and `search` request tests using the existing fake graph/index.
4. Implement JSON-line IPC with a random token stored mode `0600` under `$XDG_RUNTIME_DIR/graphify-embeddings/`.
5. Implement a `handle_request()` loop with a short timeout; call manager `tick()` between requests so idle/pressure eviction works without a monitor thread racing model inference.
6. On SIGTERM/SIGINT/stop/fatal exit, close both models and unlink socket/PID/token files.

### Task 5: CLI and Graphify pipeline integration

**Files:**
- Modify: `src/graphify_embeddings/cli.py`
- Test: `tests/test_worker.py`

**Steps:**
1. Write failing parser/dispatch tests for `worker start|stop|status|ping|unload`, `config show|path|init`, and Boolean `--rerank/--no-rerank`.
2. Test worker autostart for `index` and `search`; `--no-worker` retains the one-shot cleanup path.
3. Test config-default reranking and command-line override.
4. Add `pipeline <path>`: run Graphify first, require successful `graphify-out/graph.json`, then index, and optionally materialize semantic links. Never run embedding before Graphify completes.
5. Ensure failed Graphify leaves no model loaded and does not alter a valid existing index.
6. Update status output with model residency, device, deadline, VRAM pressure, and worker PID.

### Task 6: Dedicated uv environment and optional FlashAttention installer

**Files:**
- Create: `install.sh`
- Modify: `pyproject.toml`
- Modify: `README.md`
- Modify: `.gitignore`

**Steps:**
1. Add `nvidia-ml-py` to GPU dependencies and pin a tested compatibility range for Python 3.11, Torch, Transformers, SentenceTransformers, qwen-vl-utils, SciPy, and NumPy.
2. Implement idempotent `install.sh`: require uv, create repository `.venv` with Python 3.11, install editable GPU package, run import/version/CUDA checks, and atomically install `~/.local/bin/graphify-embeddings` symlink only after success.
3. Add `--flash-attn` option that attempts `uv pip install flash-attn --no-build-isolation`; failure must leave the base SDPA installation working and must not enable FlashAttention.
4. Add `--no-flash-attn` and document default `attention_backend = "auto"`.
5. Verify a fresh installation does not import anything from the ComfyUI venv and `sys.prefix != sys.base_prefix`.

### Task 7: Real GPU and priority gates

**Files:**
- Update: `README.md`
- Update: `plan.md`
- Update Graphify skill reference after successful verification.

**Steps:**
1. Run all unit tests, Ruff, format, Bandit, compileall, build, and Twine.
2. Install into the dedicated `.venv` and verify executable/shebang/package paths.
3. Start worker; run index on the 419-node real graph and verify embedder residency on 3090 after completion.
4. Run reranked search and verify reranker on 5090, embedder on 3090, and no same-GPU unsafe co-residency.
5. Ping at <60 seconds and prove deadlines extend; then stop pinging and prove both model PIDs/VRAM allocations disappear after >60 seconds.
6. Simulate/induce pressure with a controlled CUDA allocation and prove the worker evicts before loading another model; never interfere with ComfyUI or a game process.
7. Run `pipeline` on a small fixture repository, then load `graph.semantic.json` with real Graphify.
8. Stop worker and prove socket, PID, RAM owner process, and model VRAM allocations are gone.
9. Commit, push, and verify local HEAD equals remote `main`.
