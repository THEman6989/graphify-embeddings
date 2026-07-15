from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

from . import __version__
from .graph import GraphifyGraph
from .index import EmbeddingIndex
from .models import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_REVISION,
    DEFAULT_EMBEDDING_WRAPPER_SHA256,
    DEFAULT_INSTRUCTION,
    DEFAULT_RERANKER_MODEL,
    QwenEmbedder,
    QwenReranker,
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


def _expected_embedding_identity(args: argparse.Namespace) -> dict[str, str | None]:
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
    return {
        "model": args.embedding_model,
        "backend": "official_vl_wrapper" if official else "sentence_transformers",
        "instruction": args.instruction,
        "revision": DEFAULT_EMBEDDING_REVISION
        if args.embedding_model == DEFAULT_EMBEDDING_MODEL
        else None,
        "wrapper_sha256": wrapper_sha256,
        "dtype": str(args.dtype).lower(),
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
        "--device", default="auto", help="auto chooses cuda:0 when available"
    )
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--batch-size", type=_positive_int, default=4)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--local-files-only", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="graphify-embeddings",
        description="Local Qwen3 semantic search and reranking for Graphify graphs",
    )
    parser.add_argument("--version", action="version", version=__version__)
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
    search_parser.add_argument("--rerank", action="store_true")
    search_parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
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
    return parser


def _embedder_from_args(args: argparse.Namespace) -> QwenEmbedder:
    return QwenEmbedder(
        args.embedding_model,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        instruction=args.instruction,
        local_files_only=args.local_files_only,
    )


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


def command_index(args: argparse.Namespace) -> int:
    graph = GraphifyGraph(args.graph)
    index = EmbeddingIndex(graph)
    with _embedder_from_args(args) as embedder:
        stats = index.build(
            embedder,
            include_source=not args.no_source_context,
            force=args.force,
        )
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


def command_search(args: argparse.Namespace) -> int:
    graph = GraphifyGraph(args.graph)
    index, embedder, build_stats = _ensure_index(graph, args, keep_embedder=True)
    if embedder is None:
        raise RuntimeError("Embedding model was not initialized")
    try:
        query_vector = embedder.encode([args.query], show_progress=False)[0]
    finally:
        # Critical on a 32 GB GPU: free the 8B embedder before loading the 8B reranker.
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
    if args.json_output:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    if build_stats:
        print(
            f"Indexed {build_stats['nodes']} nodes: {build_stats['embedded']} embedded, "
            f"{build_stats['reused']} reused on {build_stats.get('device')}"
        )
    print(f"Query: {args.query}")
    print(
        f"Models: {index.metadata.get('model')}"
        + (f" -> {args.reranker_model}" if args.rerank else "")
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
        print(json.dumps(payload, indent=2, ensure_ascii=False))
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    commands = {
        "index": command_index,
        "search": command_search,
        "link": command_link,
        "info": command_info,
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
