from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol

from .config import WorkerConfig
from .gpu import GpuInfo, nvml_inventory, plan_model_devices, pressured_gpus


class ClosableModel(Protocol):
    device: str

    def close(self) -> None: ...


@dataclass
class _Slot:
    model: ClosableModel
    device: str
    deadline: float


class ModelManager:
    def __init__(
        self,
        config: WorkerConfig,
        *,
        embedder_factory: Callable[[str], ClosableModel],
        reranker_factory: Callable[[str], ClosableModel],
        inventory_provider: Callable[[], list[GpuInfo]] = nvml_inventory,
        clock: Callable[[], float] = time.monotonic,
        own_pid: int | None = None,
    ):
        self.config = config.validate()
        self._embedder_factory = embedder_factory
        self._reranker_factory = reranker_factory
        self._inventory_provider = inventory_provider
        self._clock = clock
        self._own_pid = os.getpid() if own_pid is None else own_pid
        self._lock = threading.RLock()
        self._embedding: _Slot | None = None
        self._reranker: _Slot | None = None
        self._active_requests = 0
        self._request_models: set[str] = set()
        self._pending_evictions: set[str] = set()
        self._last_pressure_check = float("-inf")

    def _deadline(self) -> float:
        return self._clock() + self.config.idle_timeout_seconds

    @staticmethod
    def _device_index(device: str) -> int | None:
        if device == "cuda":
            return 0
        if device.startswith("cuda:"):
            return int(device.split(":", 1)[1])
        return None

    def _close_slot(self, name: str, *, automatic: bool = False) -> bool:
        attribute = "_embedding" if name == "embedding" else "_reranker"
        slot = getattr(self, attribute)
        if slot is None:
            return False
        setattr(self, attribute, None)
        slot.model.close()
        if automatic:
            self._pending_evictions.add(name)
        return True

    def _target(self, name: str) -> str:
        inventory = self._inventory_provider()
        pressure = pressured_gpus(inventory, self.config, own_pid=self._own_pid)
        inventory = [gpu for gpu in inventory if gpu.index not in pressure]
        plan = plan_model_devices(
            inventory,
            self.config,
            want_embedding=name == "embedding",
            want_reranker=name == "reranker",
        )
        target = plan.embedding_device if name == "embedding" else plan.reranker_device
        if target is None:
            raise RuntimeError(f"No GPU has enough safe free VRAM for {name}")
        return target

    def _target_with_fallback(self, name: str) -> str:
        try:
            return self._target(name)
        except RuntimeError:
            other_name = "reranker" if name == "embedding" else "embedding"
            other = self._reranker if other_name == "reranker" else self._embedding
            if other is None:
                raise
            self._close_slot(other_name, automatic=True)
            return self._target(name)

    def _evict_conflict(self, name: str, target: str) -> None:
        other_name = "reranker" if name == "embedding" else "embedding"
        other = self._reranker if other_name == "reranker" else self._embedding
        if other is not None and other.device == target:
            self._close_slot(other_name, automatic=True)

    def get_embedder(self) -> ClosableModel:
        with self._lock:
            self.tick(force_pressure=True)
            if self._embedding is not None:
                self._embedding.deadline = self._deadline()
                if self._active_requests:
                    self._request_models.add("embedding")
                return self._embedding.model
            target = self._target_with_fallback("embedding")
            self._evict_conflict("embedding", target)
            model = self._embedder_factory(target)
            self._embedding = _Slot(model, target, self._deadline())
            if self._active_requests:
                self._request_models.add("embedding")
            return model

    def get_reranker(self, *, allow_disabled: bool = False) -> ClosableModel:
        with self._lock:
            if not self.config.reranker_enabled and not allow_disabled:
                raise RuntimeError("Reranker is disabled in configuration")
            self.tick(force_pressure=True)
            if self._reranker is not None:
                self._reranker.deadline = self._deadline()
                if self._active_requests:
                    self._request_models.add("reranker")
                return self._reranker.model
            target = self._target_with_fallback("reranker")
            self._evict_conflict("reranker", target)
            model = self._reranker_factory(target)
            self._reranker = _Slot(model, target, self._deadline())
            if self._active_requests:
                self._request_models.add("reranker")
            return model

    @contextmanager
    def request(self):
        with self._lock:
            if self._active_requests == 0:
                self._request_models.clear()
            self._active_requests += 1
        try:
            yield self
        finally:
            with self._lock:
                self._active_requests -= 1
                if self._active_requests == 0:
                    deadline = self._deadline()
                    if (
                        "embedding" in self._request_models
                        and self._embedding is not None
                    ):
                        self._embedding.deadline = deadline
                    if (
                        "reranker" in self._request_models
                        and self._reranker is not None
                    ):
                        self._reranker.deadline = deadline
                    self._request_models.clear()

    def ping(self, models: Iterable[str] = ("embedding", "reranker")) -> list[str]:
        with self._lock:
            kept = []
            for name in models:
                if name not in {"embedding", "reranker"}:
                    raise ValueError(f"Unknown model name: {name}")
                slot = self._embedding if name == "embedding" else self._reranker
                if slot is not None:
                    slot.deadline = self._deadline()
                    kept.append(name)
            return kept

    def tick(self, *, force_pressure: bool = False) -> list[str]:
        with self._lock:
            if self._active_requests:
                return []
            now = self._clock()
            evicted = []
            if self._embedding is not None and now >= self._embedding.deadline:
                self._close_slot("embedding", automatic=True)
                evicted.append("embedding")
            if self._reranker is not None and now >= self._reranker.deadline:
                self._close_slot("reranker", automatic=True)
                evicted.append("reranker")
            due = now - self._last_pressure_check >= self.config.pressure_poll_seconds
            if force_pressure or due:
                inventory = self._inventory_provider()
                pressure = pressured_gpus(inventory, self.config, own_pid=self._own_pid)
                self._last_pressure_check = now
                for name, slot in (
                    ("embedding", self._embedding),
                    ("reranker", self._reranker),
                ):
                    if slot is None:
                        continue
                    index = self._device_index(slot.device)
                    if index is not None and index in pressure:
                        self._close_slot(name, automatic=True)
                        evicted.append(name)
            return evicted

    def consume_evictions(self) -> list[str]:
        with self._lock:
            evicted = sorted(self._pending_evictions)
            self._pending_evictions.clear()
            return evicted

    def unload(self, models: Iterable[str] = ("embedding", "reranker")) -> list[str]:
        with self._lock:
            unloaded = []
            for name in models:
                if name not in {"embedding", "reranker"}:
                    raise ValueError(f"Unknown model name: {name}")
                if self._close_slot(name):
                    unloaded.append(name)
            return unloaded

    def status(self) -> dict:
        with self._lock:

            def describe(slot: _Slot | None):
                if slot is None:
                    return None
                return {
                    "device": slot.device,
                    "deadline_monotonic": slot.deadline,
                    "idle_seconds_remaining": max(0.0, slot.deadline - self._clock()),
                }

            return {
                "embedding": describe(self._embedding),
                "reranker": describe(self._reranker),
                "active_requests": self._active_requests,
                "idle_timeout_seconds": self.config.idle_timeout_seconds,
            }

    def close(self) -> None:
        with self._lock:
            self._close_slot("embedding")
            self._close_slot("reranker")
