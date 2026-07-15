from __future__ import annotations

import os
import hashlib
import json
import math
import re
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


_DEVICE_RE = re.compile(r"(?:auto|cpu|cuda(?::\d+)?)")


def default_config_path() -> Path:
    root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return root / "graphify-embeddings" / "config.toml"


def default_runtime_dir() -> Path:
    value = os.environ.get("XDG_RUNTIME_DIR")
    root = Path(value) if value else Path.home() / ".cache"
    return root / "graphify-embeddings"


@dataclass(frozen=True)
class WorkerConfig:
    worker_enabled: bool = True
    auto_start: bool = True
    idle_timeout_seconds: float = 60.0
    pressure_poll_seconds: float = 2.0
    reranker_enabled: bool = True
    placement_policy: str = "split"
    preferred_gpu: int = 0
    secondary_gpu: int = 1
    embedding_device: str = "auto"
    reranker_device: str = "auto"
    min_free_gib: float = 6.0
    high_priority_process_mib: int = 2048
    embedding_required_gib: float = 16.0
    reranker_required_gib: float = 17.0
    attention_backend: str = "auto"

    def validate(self) -> "WorkerConfig":
        if (
            not math.isfinite(self.idle_timeout_seconds)
            or self.idle_timeout_seconds <= 0
        ):
            raise ValueError("idle_timeout_seconds must be greater than zero")
        if (
            not math.isfinite(self.pressure_poll_seconds)
            or self.pressure_poll_seconds <= 0
        ):
            raise ValueError("pressure_poll_seconds must be greater than zero")
        if self.placement_policy not in {"split", "sequential"}:
            raise ValueError("placement_policy must be 'split' or 'sequential'")
        if self.preferred_gpu < 0 or self.secondary_gpu < 0:
            raise ValueError("GPU indices must be non-negative")
        for name in ("embedding_device", "reranker_device"):
            value = str(getattr(self, name)).lower()
            if not _DEVICE_RE.fullmatch(value):
                raise ValueError(f"{name} has invalid device syntax: {value}")
        if not math.isfinite(self.min_free_gib) or self.min_free_gib < 0:
            raise ValueError(
                "GPU GiB values must be finite and min_free_gib non-negative"
            )
        if self.high_priority_process_mib < 0:
            raise ValueError("high_priority_process_mib must be non-negative")
        if (
            not math.isfinite(self.embedding_required_gib)
            or not math.isfinite(self.reranker_required_gib)
            or self.embedding_required_gib <= 0
            or self.reranker_required_gib <= 0
        ):
            raise ValueError(
                "model required GiB values must be finite and greater than zero"
            )
        if self.attention_backend not in {"auto", "sdpa", "flash_attention_2"}:
            raise ValueError(
                "attention_backend must be auto, sdpa, or flash_attention_2"
            )
        return self


def config_fingerprint(config: WorkerConfig) -> str:
    payload = {
        item.name: getattr(config, item.name)
        for item in fields(config)
        if item.name not in {"worker_enabled", "auto_start"}
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _flatten(data: dict[str, Any]) -> dict[str, Any]:
    worker = data.get("worker", {})
    gpu = data.get("gpu", {})
    models = data.get("models", {})
    attention = data.get("attention", {})
    return {
        "worker_enabled": worker.get("enabled"),
        "auto_start": worker.get("auto_start"),
        "idle_timeout_seconds": worker.get("idle_timeout_seconds"),
        "pressure_poll_seconds": worker.get("pressure_poll_seconds"),
        "reranker_enabled": worker.get("reranker_enabled"),
        "placement_policy": gpu.get("placement_policy"),
        "preferred_gpu": gpu.get("preferred_gpu"),
        "secondary_gpu": gpu.get("secondary_gpu"),
        "embedding_device": gpu.get("embedding_device"),
        "reranker_device": gpu.get("reranker_device"),
        "min_free_gib": gpu.get("min_free_gib"),
        "high_priority_process_mib": gpu.get("high_priority_process_mib"),
        "embedding_required_gib": models.get("embedding_required_gib"),
        "reranker_required_gib": models.get("reranker_required_gib"),
        "attention_backend": attention.get("backend"),
    }


def load_config(path: str | Path | None = None) -> WorkerConfig:
    config_path = Path(path).expanduser() if path is not None else default_config_path()
    if not config_path.is_file():
        return WorkerConfig().validate()
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    known = {item.name for item in fields(WorkerConfig)}
    overrides = {
        key: value
        for key, value in _flatten(raw).items()
        if key in known and value is not None
    }
    return WorkerConfig(**overrides).validate()
