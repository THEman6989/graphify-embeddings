from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .client import RuntimePaths, WorkerClient, WorkerError, ensure_worker
from .config import WorkerConfig, default_config_path, load_config
from .graph import DOCUMENT_SCHEMA, GraphifyGraph
from .index import EmbeddingIndex
from .models import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_REVISION,
    DEFAULT_EMBEDDING_WRAPPER_SHA256,
    DEFAULT_INSTRUCTION,
    DEFAULT_RERANKER_MODEL,
    QwenEmbedder,
    QwenReranker,
    local_model_fingerprint,
    resolve_attention_backend,
    resolve_model_revision,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _cosine_threshold(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not -1.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be finite and within [-1, 1]")
    return parsed


def _expected_embedding_identity(args: argparse.Namespace) -> dict[str, Any]:
    local = Path(args.embedding_model).expanduser()
    script = local / "scripts" / "qwen3_vl_embedding.py" if local.is_dir() else None
    wrapper_sha256 = (
        hashlib.sha256(script.read_bytes()).hexdigest()
        if script and script.is_file()
        else None
    )
    if args.embedding_model == DEFAULT_EMBEDDING_MODEL:
        wrapper_sha256 = DEFAULT_EMBEDDING_WRAPPER_SHA256
    official = (
        args.embedding_model == DEFAULT_EMBEDDING_MODEL or wrapper_sha256 is not None
    )
    revision = resolve_model_revision(
        args.embedding_model,
        args.embedding_revision,
        default_model=DEFAULT_EMBEDDING_MODEL,
        default_revision=DEFAULT_EMBEDDING_REVISION,
    )
    artifact_fingerprint = local_model_fingerprint(local) if local.is_dir() else None
    return {
        "model": args.embedding_model,
        "backend": "official_vl_wrapper" if official else "sentence_transformers",
        "instruction": args.instruction,
        "revision": revision,
        "wrapper_sha256": wrapper_sha256,
        "artifact_fingerprint": artifact_fingerprint,
        "dtype": str(args.dtype).lower(),
        "attention_backend": resolve_attention_backend(args.attention_backend),
        "document_schema": DOCUMENT_SCHEMA,
    }


def _add_graph_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--graph",
        default="graphify-out/graph.json",
        help="Path to Graphify graph.json (default: graphify-out/graph.json)",
    )


def _add_embedding_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument(
        "--embedding-revision",
        help="Immutable 40-character commit for an alternative remote embedding model",
    )
    parser.add_argument(
        "--device", default="auto", help="auto chooses cuda:0 when available"
    )
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--batch-size", type=_positive_int, default=4)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--attention-backend",
        choices=["auto", "sdpa", "flash_attention_2"],
        default="auto",
    )
    parser.add_argument(
        "--no-worker", action="store_true", help="Use one-shot model loading"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="graphify-embeddings",
        description="Local Qwen3 semantic search and reranking for Graphify graphs",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--config", help="Path to config.toml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser(
        "index", help="Embed Graphify nodes incrementally"
    )
    _add_graph_argument(index_parser)
    _add_embedding_arguments(index_parser)
    index_parser.add_argument("--force", action="store_true")
    index_parser.add_argument("--no-source-context", action="store_true")

    search_parser = subparsers.add_parser(
        "search", help="Semantic/hybrid search over a Graphify graph"
    )
    search_parser.add_argument("query")
    _add_graph_argument(search_parser)
    _add_embedding_arguments(search_parser)
    search_parser.add_argument("--top-k", type=_positive_int, default=10)
    search_parser.add_argument("--candidate-k", type=_positive_int, default=24)
    search_parser.add_argument("--neighbors", type=_nonnegative_int, default=1)
    search_parser.add_argument(
        "--rerank",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable reranking; default comes from config.toml",
    )
    search_parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    search_parser.add_argument(
        "--reranker-revision",
        help="Immutable 40-character commit for an alternative remote reranker",
    )
    search_parser.add_argument("--reranker-batch-size", type=_positive_int, default=1)
    search_parser.add_argument("--json", action="store_true", dest="json_output")
    search_parser.add_argument("--force-index", action="store_true")
    search_parser.add_argument("--no-source-context", action="store_true")

    link_parser = subparsers.add_parser(
        "link", help="Add semantically_similar_to edges"
    )
    _add_graph_argument(link_parser)
    _add_embedding_arguments(link_parser)
    link_parser.add_argument("--threshold", type=_cosine_threshold, default=0.82)
    link_parser.add_argument("--max-neighbors", type=_positive_int, default=5)
    link_parser.add_argument("--block-size", type=_positive_int, default=512)
    link_parser.add_argument("--output")
    link_parser.add_argument("--in-place", action="store_true")
    link_parser.add_argument("--force-index", action="store_true")
    link_parser.add_argument("--no-source-context", action="store_true")

    info_parser = subparsers.add_parser("info", help="Show graph and index status")
    _add_graph_argument(info_parser)
    info_parser.add_argument("--json", action="store_true", dest="json_output")

    worker_parser = subparsers.add_parser(
        "worker", help="Manage the local persistent model worker"
    )
    worker_actions = worker_parser.add_subparsers(dest="worker_action", required=True)
    worker_actions.add_parser("start", help="Start the local model worker")
    worker_actions.add_parser("stop", help="Stop the worker and unload all models")
    worker_actions.add_parser("status", help="Show worker/model residency")
    for action in ("ping", "warm", "unload"):
        action_parser = worker_actions.add_parser(action)
        action_parser.add_argument(
            "--models",
            nargs="+",
            choices=["embedding", "reranker"],
            default=["embedding", "reranker"],
        )

    pipeline_parser = subparsers.add_parser(
        "pipeline", help="Run Graphify, then embedding index and semantic linking"
    )
    pipeline_parser.add_argument("path", nargs="?", default=".")
    pipeline_parser.add_argument(
        "--rerank", action=argparse.BooleanOptionalAction, default=None
    )
    pipeline_parser.add_argument(
        "--link", action=argparse.BooleanOptionalAction, default=True
    )
    pipeline_parser.add_argument("--code-only", action="store_true")

    config_parser = subparsers.add_parser("config", help="Manage config.toml")
    config_actions = config_parser.add_subparsers(dest="config_action", required=True)
    config_actions.add_parser("path")
    config_actions.add_parser("show")
    init_parser = config_actions.add_parser("init")
    init_parser.add_argument("--force", action="store_true")
    return parser


def _worker_compatible(args: argparse.Namespace, config: WorkerConfig) -> bool:
    return bool(
        config.worker_enabled
        and not getattr(args, "no_worker", False)
        and not getattr(args, "local_files_only", False)
        and getattr(args, "embedding_model", DEFAULT_EMBEDDING_MODEL)
        == DEFAULT_EMBEDDING_MODEL
        and getattr(args, "embedding_revision", None) is None
        and getattr(args, "device", "auto") == "auto"
        and getattr(args, "dtype", "bf16") == "bf16"
        and getattr(args, "instruction", DEFAULT_INSTRUCTION) == DEFAULT_INSTRUCTION
        and getattr(args, "attention_backend", "auto")
        in {"auto", config.attention_backend}
    )


def _worker_client(
    args: argparse.Namespace, config: WorkerConfig
) -> WorkerClient | None:
    if config.auto_start:
        return ensure_worker(config_path=getattr(args, "config", None))
    client = WorkerClient(RuntimePaths.defaults())
    try:
        client.request("status")
    except WorkerError:
        return None
    return client


def _embedder_from_args(args: argparse.Namespace) -> QwenEmbedder:
    return QwenEmbedder(
        args.embedding_model,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        instruction=args.instruction,
        local_files_only=args.local_files_only,
        revision=args.embedding_revision,
        attention_backend=args.attention_backend,
    )


def _apply_attention_config(args: argparse.Namespace, config: WorkerConfig) -> None:
    if getattr(args, "attention_backend", "auto") == "auto":
        args.attention_backend = config.attention_backend


def _ensure_index(
    graph: GraphifyGraph,
    args: argparse.Namespace,
    *,
    keep_embedder: bool,
) -> tuple[EmbeddingIndex, QwenEmbedder | None, dict | None]:
    index = EmbeddingIndex(graph)
    include_source = not args.no_source_context
    needs_build = args.force_index if hasattr(args, "force_index") else args.force
    if index.exists() and not needs_build:
        try:
            index.load()
            current_hashes = graph.content_hashes(include_source=include_source)
            expected_identity = _expected_embedding_identity(args)
            stored_identity = index.metadata.get("embedding_identity", {})
            identity_mismatch = any(
                stored_identity.get(key) != value
                for key, value in expected_identity.items()
                if key != "wrapper_sha256" or value is not None
            )
            needs_build = (
                identity_mismatch
                or index.metadata.get("include_source") != include_source
                or set(index.node_ids) != set(graph.by_id)
                or index.metadata.get("content_hashes", {}) != current_hashes
            )
        except Exception:
            needs_build = True
    else:
        needs_build = True

    embedder: QwenEmbedder | None = None
    stats = None
    if needs_build or keep_embedder:
        embedder = _embedder_from_args(args)
    if needs_build:
        if embedder is None:
            raise RuntimeError("Embedding model was not initialized")
        try:
            stats = index.build(
                embedder,
                include_source=include_source,
                force=bool(
                    getattr(args, "force_index", False) or getattr(args, "force", False)
                ),
            )
        except Exception:
            if embedder is not None:
                embedder.close()
                embedder = None
            raise
    if not keep_embedder and embedder is not None:
        embedder.close()
        embedder = None
    return index, embedder, stats


def _index_stats(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    _apply_attention_config(args, config)
    client = None
    if _worker_compatible(args, config):
        client = _worker_client(args, config)
    if client is not None:
        return client.request(
            "index",
            {
                "graph": str(Path(args.graph).expanduser().resolve()),
                "force": args.force,
                "include_source": not args.no_source_context,
            },
        )
    graph = GraphifyGraph(args.graph)
    index = EmbeddingIndex(graph)
    with _embedder_from_args(args) as embedder:
        return index.build(
            embedder,
            include_source=not args.no_source_context,
            force=args.force,
        )


def command_index(args: argparse.Namespace) -> int:
    stats = _index_stats(args)
    print(json.dumps(stats, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


def command_search(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    _apply_attention_config(args, config)
    args.rerank = config.reranker_enabled if args.rerank is None else args.rerank
    client = None
    if _worker_compatible(args, config) and (
        args.reranker_model == DEFAULT_RERANKER_MODEL and args.reranker_revision is None
    ):
        client = _worker_client(args, config)
    if client is not None:
        payload = client.request(
            "search",
            {
                "graph": str(Path(args.graph).expanduser().resolve()),
                "query": args.query,
                "top_k": args.top_k,
                "candidate_k": args.candidate_k,
                "neighbors": args.neighbors,
                "rerank": args.rerank,
                "reranker_batch_size": args.reranker_batch_size,
                "force_index": args.force_index,
                "include_source": not args.no_source_context,
            },
        )
    else:
        graph = GraphifyGraph(args.graph)
        index, embedder, build_stats = _ensure_index(graph, args, keep_embedder=True)
        if embedder is None:
            raise RuntimeError("Embedding model was not initialized")
        try:
            query_vector = embedder.encode([args.query], show_progress=False)[0]
        finally:
            embedder.close()

        reranker = None
        try:
            if args.rerank:
                reranker = QwenReranker(
                    args.reranker_model,
                    device=args.device,
                    dtype=args.dtype,
                    instruction=args.instruction,
                    local_files_only=args.local_files_only,
                    revision=args.reranker_revision,
                    attention_backend=args.attention_backend,
                )
            results = index.search(
                args.query,
                query_vector,
                top_k=args.top_k,
                candidate_k=args.candidate_k,
                neighbors=args.neighbors,
                reranker=reranker,
                rerank_batch_size=args.reranker_batch_size,
            )
        finally:
            if reranker is not None:
                reranker.close()

        payload = {
            "query": args.query,
            "graph": str(graph.path),
            "embedding_model": index.metadata.get("model"),
            "reranker_model": args.reranker_model if args.rerank else None,
            "index_build": build_stats,
            "results": results,
        }
    build_stats = payload.get("index_build")
    results = payload["results"]
    if args.json_output:
        print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))
        return 0

    if build_stats:
        print(
            f"Indexed {build_stats['nodes']} nodes: {build_stats['embedded']} embedded, "
            f"{build_stats['reused']} reused on {build_stats.get('device')}"
        )
    print(f"Query: {args.query}")
    print(
        f"Models: {payload.get('embedding_model')}"
        + (f" -> {payload.get('reranker_model')}" if args.rerank else "")
    )
    for rank, item in enumerate(results, 1):
        location = ":".join(
            part
            for part in (
                str(item.get("source_file") or ""),
                str(item.get("source_location") or ""),
            )
            if part
        )
        extra = (
            f" rerank={item['reranker_score']:.4f}" if "reranker_score" in item else ""
        )
        print(
            f"{rank:>2}. {item['label']}  score={item['score']:.4f} "
            f"semantic={item['semantic_score']:.4f}{extra}  {location}"
        )
        for neighbor in item.get("neighbors", [])[:8]:
            neighbor_location = ":".join(
                part
                for part in (
                    str(neighbor.get("source_file") or ""),
                    str(neighbor.get("source_location") or ""),
                )
                if part
            )
            print(
                f"      -> [{neighbor.get('relation')}] {neighbor.get('label')} "
                f"({neighbor_location})"
            )
    return 0


def command_link(args: argparse.Namespace) -> int:
    if args.in_place and args.output:
        raise ValueError("--in-place and --output are mutually exclusive")
    if (
        args.output
        and Path(args.output).expanduser().resolve()
        == Path(args.graph).expanduser().resolve()
    ):
        raise ValueError(
            "--output cannot overwrite --graph; use --in-place for backup protection"
        )
    _apply_attention_config(args, load_config(args.config))
    graph = GraphifyGraph(args.graph)
    index, _, build_stats = _ensure_index(graph, args, keep_embedder=False)
    pairs = index.similarity_pairs(
        threshold=args.threshold,
        max_neighbors=args.max_neighbors,
        block_size=args.block_size,
    )
    target, count = index.write_linked_graph(
        pairs,
        in_place=args.in_place,
        output=args.output,
    )
    print(
        json.dumps(
            {
                "graph": str(target),
                "semantic_edges": count,
                "threshold": args.threshold,
                "model": index.metadata.get("model"),
                "index_build": build_stats,
                "backup": str(graph.path) + ".bak" if args.in_place else None,
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


def command_info(args: argparse.Namespace) -> int:
    graph = GraphifyGraph(args.graph)
    index = EmbeddingIndex(graph)
    payload = {
        "graph": str(graph.path),
        "nodes": len(graph.nodes),
        "links": len(graph.links),
        "directed": graph.directed,
        "built_at_commit": graph.data.get("built_at_commit"),
        "index_exists": index.exists(),
    }
    if index.exists():
        index.load()
        payload["index"] = index.metadata
    if args.json_output:
        print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))
    else:
        print(f"Graph: {payload['graph']}")
        print(
            f"Nodes: {payload['nodes']}  Links: {payload['links']}  Directed: {payload['directed']}"
        )
        print(f"Embedding index: {'ready' if payload['index_exists'] else 'missing'}")
        if payload.get("index"):
            print(
                f"Model: {payload['index'].get('model')}  "
                f"Dim: {payload['index'].get('dimension')}  "
                f"Nodes: {payload['index'].get('node_count')}"
            )
    return 0


DEFAULT_CONFIG_TEXT = """[worker]
enabled = true
auto_start = true
idle_timeout_seconds = 60
pressure_poll_seconds = 2
reranker_enabled = true

[gpu]
placement_policy = "split"
preferred_gpu = 0
secondary_gpu = 1
embedding_device = "auto"
reranker_device = "auto"
min_free_gib = 6
high_priority_process_mib = 2048

[models]
embedding_required_gib = 16
reranker_required_gib = 17

[attention]
backend = "auto"
"""


def command_config(args: argparse.Namespace) -> int:
    path = Path(args.config).expanduser() if args.config else default_config_path()
    if args.config_action == "path":
        print(path)
        return 0
    if args.config_action == "init":
        if path.exists() and not args.force:
            raise FileExistsError(f"Config already exists: {path}; use --force")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(DEFAULT_CONFIG_TEXT, encoding="utf-8")
        temporary.replace(path)
        print(path)
        return 0
    config = load_config(path)
    print(json.dumps(config.__dict__, indent=2, ensure_ascii=False))
    return 0


def command_worker(args: argparse.Namespace) -> int:
    if args.worker_action == "start":
        client = ensure_worker(config_path=args.config)
        result = client.request("status")
    else:
        client = WorkerClient(RuntimePaths.defaults())
        payload = {}
        if args.worker_action in {"ping", "warm", "unload"}:
            payload["models"] = args.models
        result = client.request(args.worker_action, payload)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


def command_pipeline(args: argparse.Namespace) -> int:
    project = Path(args.path).expanduser().resolve()
    if not project.is_dir():
        raise FileNotFoundError(f"Project directory not found: {project}")
    graphify_command = ["graphify", "extract", "."]
    if args.code_only:
        graphify_command.append("--code-only")
    completed = subprocess.run(
        graphify_command, cwd=project, text=True, capture_output=True
    )
    used_code_only_fallback = False
    if completed.returncode and not args.code_only:
        completed = subprocess.run(
            graphify_command + ["--code-only"],
            cwd=project,
            text=True,
            capture_output=True,
        )
        used_code_only_fallback = completed.returncode == 0
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"Graphify extraction failed: {detail}")
    graph_path = project / "graphify-out" / "graph.json"
    if not graph_path.is_file():
        raise FileNotFoundError(f"Graphify did not create {graph_path}")

    config = load_config(args.config)
    client = _worker_client(args, config) if config.worker_enabled else None
    if client is not None:
        index_stats = client.request(
            "index",
            {"graph": str(graph_path), "force": False, "include_source": True},
        )
        rerank = config.reranker_enabled if args.rerank is None else args.rerank
        warm_models = ["embedding"] + (["reranker"] if rerank else [])
        residency = client.request("warm", {"models": warm_models})
    else:
        index_args = build_parser().parse_args(
            ["index", "--graph", str(graph_path), "--no-worker"]
        )
        index_args.config = args.config
        index_args.attention_backend = config.attention_backend
        index_stats = {**_index_stats(index_args), "worker": False}
        residency = None

    semantic_graph = None
    semantic_edges = None
    if args.link:
        graph = GraphifyGraph(graph_path)
        index = EmbeddingIndex(graph).load()
        pairs = index.similarity_pairs(threshold=0.82, max_neighbors=5, block_size=512)
        semantic_graph, semantic_edges = index.write_linked_graph(pairs)
    result = {
        "project": str(project),
        "graph": str(graph_path),
        "graphify_code_only_fallback": used_code_only_fallback,
        "index": index_stats,
        "semantic_graph": str(semantic_graph) if semantic_graph else None,
        "semantic_edges": semantic_edges,
        "residency": residency,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    commands = {
        "index": command_index,
        "search": command_search,
        "link": command_link,
        "info": command_info,
        "worker": command_worker,
        "pipeline": command_pipeline,
        "config": command_config,
    }
    try:
        return commands[args.command](args)
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
