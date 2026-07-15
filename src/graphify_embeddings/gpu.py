from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .config import WorkerConfig


@dataclass(frozen=True)
class GpuProcess:
    pid: int
    name: str
    used_memory_mib: int


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    total_gib: float
    free_gib: float
    processes: tuple[GpuProcess, ...]


@dataclass(frozen=True)
class PlacementPlan:
    embedding_device: str | None
    reranker_device: str | None

    @property
    def same_gpu(self) -> bool:
        return (
            self.embedding_device is not None
            and self.embedding_device == self.reranker_device
        )


def _explicit_index(device: str) -> int | None:
    if device == "auto":
        return None
    if device == "cuda":
        return 0
    if device.startswith("cuda:"):
        return int(device.split(":", 1)[1])
    return None


def _fits(free_gib: float, required_gib: float, reserve_gib: float) -> bool:
    return free_gib >= required_gib + reserve_gib


def plan_model_devices(
    gpus: Iterable[GpuInfo],
    config: WorkerConfig,
    *,
    want_embedding: bool = True,
    want_reranker: bool | None = None,
) -> PlacementPlan:
    config.validate()
    by_index = {gpu.index: gpu for gpu in gpus}
    if want_reranker is None:
        want_reranker = config.reranker_enabled
    free = {index: gpu.free_gib for index, gpu in by_index.items()}

    def allocate(
        requested: str,
        required: float,
        candidates: list[int],
    ) -> str | None:
        if requested == "cpu":
            return "cpu"
        explicit = _explicit_index(requested)
        ordered = [explicit] if explicit is not None else candidates
        for index in ordered:
            if index not in by_index:
                if explicit is not None:
                    raise RuntimeError(f"Configured CUDA device {index} is unavailable")
                continue
            if _fits(free[index], required, config.min_free_gib):
                free[index] -= required
                return f"cuda:{index}"
        if explicit is not None:
            raise RuntimeError(
                f"Configured CUDA device {explicit} lacks safe free VRAM for {required:.2f} GiB"
            )
        return None

    all_indices = sorted(by_index)
    preferred = [config.preferred_gpu] + [
        index for index in all_indices if index != config.preferred_gpu
    ]
    secondary = [config.secondary_gpu] + [
        index for index in preferred if index != config.secondary_gpu
    ]
    reranker_device = None
    if want_reranker:
        reranker_device = allocate(
            config.reranker_device,
            config.reranker_required_gib,
            preferred,
        )
    embedding_device = None
    if want_embedding:
        embedding_candidates = (
            secondary if config.placement_policy == "split" else preferred
        )
        embedding_device = allocate(
            config.embedding_device,
            config.embedding_required_gib,
            embedding_candidates,
        )
    return PlacementPlan(embedding_device, reranker_device)


def pressured_gpus(
    gpus: Iterable[GpuInfo], config: WorkerConfig, *, own_pid: int | None = None
) -> set[int]:
    own_pid = os.getpid() if own_pid is None else own_pid
    result: set[int] = set()
    for gpu in gpus:
        if gpu.free_gib < config.min_free_gib:
            result.add(gpu.index)
            continue
        if any(
            process.pid != own_pid
            and process.used_memory_mib >= config.high_priority_process_mib
            for process in gpu.processes
        ):
            result.add(gpu.index)
    return result


def _process_name(pid: int) -> str:
    try:
        raw = (Path("/proc") / str(pid) / "cmdline").read_bytes()
        text = raw.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
        return text or str(pid)
    except OSError:
        return str(pid)


def nvml_inventory() -> list[GpuInfo]:
    try:
        import pynvml
    except ImportError as exc:
        raise RuntimeError(
            "nvidia-ml-py is required for automatic GPU placement and pressure eviction"
        ) from exc
    pynvml.nvmlInit()
    try:
        result = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            process_map: dict[int, GpuProcess] = {}
            readers: list[Callable] = [pynvml.nvmlDeviceGetComputeRunningProcesses]
            graphics_reader = getattr(
                pynvml, "nvmlDeviceGetGraphicsRunningProcesses", None
            )
            if graphics_reader is not None:
                readers.append(graphics_reader)
            for reader in readers:
                try:
                    processes = reader(handle)
                except pynvml.NVMLError:
                    continue
                for process in processes:
                    used = getattr(process, "usedGpuMemory", 0)
                    if used is None or used < 0:
                        used = 0
                    process_map[process.pid] = GpuProcess(
                        process.pid,
                        _process_name(process.pid),
                        int(used / 1024**2),
                    )
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            result.append(
                GpuInfo(
                    index=index,
                    name=str(name),
                    total_gib=memory.total / 1024**3,
                    free_gib=memory.free / 1024**3,
                    processes=tuple(process_map.values()),
                )
            )
        return result
    finally:
        pynvml.nvmlShutdown()
