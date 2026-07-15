from __future__ import annotations

import hashlib
import json
import re
from collections import deque
from pathlib import Path
from typing import Any, Iterable


_LINE_RE = re.compile(r"L(\d+)")
_TOKEN_RE = re.compile(r"[A-Za-z_ÄÖÜäöüß][A-Za-z0-9_ÄÖÜäöüß.-]{1,}")


class GraphFormatError(ValueError):
    pass


class GraphifyGraph:
    """Validated, lightweight view of a Graphify node-link JSON graph."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"Graphify graph not found: {self.path}")
        self.data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(self.data, dict):
            raise GraphFormatError("Graph JSON must be an object")
        if not isinstance(self.data.get("nodes"), list):
            raise GraphFormatError("Graph JSON has no nodes list")
        if not isinstance(self.data.get("links", []), list):
            raise GraphFormatError("Graph JSON links must be a list")

        self.nodes: list[dict[str, Any]] = []
        self.by_id: dict[str, dict[str, Any]] = {}
        for raw in self.data["nodes"]:
            if not isinstance(raw, dict) or raw.get("id") is None:
                continue
            node = raw
            node_id = str(node["id"])
            if node_id in self.by_id:
                continue
            self.nodes.append(node)
            self.by_id[node_id] = node

        self.links: list[dict[str, Any]] = [
            link for link in self.data.get("links", []) if isinstance(link, dict)
        ]
        self.directed = bool(self.data.get("directed", False))
        self.root = self.path.parent.parent
        self._adjacency = self._build_adjacency()

    def _build_adjacency(self) -> dict[str, list[tuple[str, dict[str, Any]]]]:
        adjacency: dict[str, list[tuple[str, dict[str, Any]]]] = {
            node_id: [] for node_id in self.by_id
        }
        for link in self.links:
            source = str(link.get("source", ""))
            target = str(link.get("target", ""))
            if source not in adjacency or target not in adjacency:
                continue
            adjacency[source].append((target, link))
            if not self.directed:
                adjacency[target].append((source, link))
        return adjacency

    @staticmethod
    def _flatten_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (int, float, bool)):
            return str(value)
        if isinstance(value, list):
            return " ".join(GraphifyGraph._flatten_text(v) for v in value).strip()
        if isinstance(value, dict):
            return " ".join(
                f"{key}: {GraphifyGraph._flatten_text(val)}"
                for key, val in value.items()
                if key not in {"embedding", "vector"}
            ).strip()
        return str(value).strip()

    def source_context(self, node: dict[str, Any], radius: int = 12) -> str:
        source = str(node.get("source_file") or "").strip()
        location = str(node.get("source_location") or "")
        match = _LINE_RE.search(location)
        if not source or not match:
            return ""
        source_path = (self.root / source).resolve()
        try:
            source_path.relative_to(self.root)
        except ValueError:
            return ""
        if not source_path.is_file():
            return ""
        try:
            lines = source_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return ""
        line_index = max(0, int(match.group(1)) - 1)
        start = max(0, line_index - radius)
        stop = min(len(lines), line_index + radius + 1)
        return "\n".join(lines[start:stop])

    def node_text(
        self,
        node: dict[str, Any],
        *,
        include_source: bool = True,
        max_chars: int = 12_000,
    ) -> str:
        preferred = (
            "label",
            "kind",
            "type",
            "entity_type",
            "description",
            "docstring",
            "summary",
            "content",
            "source_file",
            "source_location",
        )
        parts: list[str] = []
        for key in preferred:
            text = self._flatten_text(node.get(key))
            if text:
                parts.append(f"{key}: {text}")
        if include_source:
            context = self.source_context(node)
            if context:
                parts.append(f"source_context:\n{context}")
        if not parts:
            parts.append(f"id: {node.get('id', '')}")
        return "\n".join(parts)[:max_chars]

    def documents(self, *, include_source: bool = True) -> list[tuple[str, str]]:
        return [
            (str(node["id"]), self.node_text(node, include_source=include_source))
            for node in self.nodes
        ]

    def content_hashes(self, *, include_source: bool = True) -> dict[str, str]:
        return {
            node_id: hashlib.sha256(text.encode("utf-8")).hexdigest()
            for node_id, text in self.documents(include_source=include_source)
        }

    def degree(self, node_id: str) -> int:
        return len(self._adjacency.get(node_id, ()))

    def neighbors(self, node_id: str, depth: int = 1, limit: int = 30) -> list[dict[str, Any]]:
        if depth <= 0 or node_id not in self.by_id:
            return []
        seen = {node_id}
        queue: deque[tuple[str, int]] = deque([(node_id, 0)])
        found: list[dict[str, Any]] = []
        while queue and len(found) < limit:
            current, current_depth = queue.popleft()
            if current_depth >= depth:
                continue
            for target, link in self._adjacency.get(current, ()):
                if target in seen:
                    continue
                seen.add(target)
                target_node = self.by_id[target]
                found.append(
                    {
                        "id": target,
                        "label": target_node.get("label", target),
                        "source_file": target_node.get("source_file"),
                        "source_location": target_node.get("source_location"),
                        "relation": link.get("relation") or link.get("type"),
                        "confidence": link.get("confidence"),
                        "depth": current_depth + 1,
                    }
                )
                queue.append((target, current_depth + 1))
                if len(found) >= limit:
                    break
        return found

    @staticmethod
    def lexical_score(query: str, text: str) -> float:
        query_tokens = {token.lower() for token in _TOKEN_RE.findall(query)}
        if not query_tokens:
            return 0.0
        text_tokens = {token.lower() for token in _TOKEN_RE.findall(text)}
        return len(query_tokens & text_tokens) / len(query_tokens)

    def semantic_links(
        self,
        pairs: Iterable[tuple[str, str, float]],
        *,
        model: str,
    ) -> list[dict[str, Any]]:
        links = []
        for source, target, score in pairs:
            links.append(
                {
                    "source": source,
                    "target": target,
                    "relation": "semantically_similar_to",
                    "type": "semantically_similar_to",
                    "context": "qwen_embedding_cosine",
                    "confidence": "INFERRED",
                    "confidence_score": round(float(score), 6),
                    "weight": round(float(score), 6),
                    "embedding_model": model,
                }
            )
        return links
