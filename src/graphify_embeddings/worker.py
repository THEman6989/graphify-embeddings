from __future__ import annotations

import argparse
import contextlib
import fcntl
import hmac
import json
import os
import secrets
import socket
import socketserver
import signal
import struct
import stat
from pathlib import Path
from typing import Any, Callable

from .client import MAX_MESSAGE_BYTES, RuntimePaths
from .config import config_fingerprint, load_config
from .manager import ModelManager
from .models import QwenEmbedder, QwenReranker
from .service import ModelService


Operation = Callable[[dict[str, Any]], Any]


def _peer_uid(connection) -> int:
    if not hasattr(socket, "SO_PEERCRED"):
        raise RuntimeError("SO_PEERCRED is required for worker authentication")
    credentials = connection.getsockopt(
        socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")
    )
    _pid, uid, _gid = struct.unpack("3i", credentials)
    return uid


class _RequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        owner: WorkerServer = self.server.owner  # type: ignore[attr-defined]
        try:
            if _peer_uid(self.request) != os.getuid():
                raise PermissionError("Worker peer UID does not match the owner")
        except Exception as exc:
            owner._respond(self.wfile, False, error=str(exc))
            return
        self.request.settimeout(owner.read_timeout)
        try:
            raw = self.rfile.readline(MAX_MESSAGE_BYTES + 1)
        except (socket.timeout, TimeoutError):
            return
        if not raw:
            return
        if len(raw) > MAX_MESSAGE_BYTES:
            owner._respond(self.wfile, False, error="Request exceeds the size limit")
            return
        try:
            request = json.loads(raw)
            if not isinstance(request, dict):
                raise ValueError("Request must be a JSON object")
            token = request.get("token")
            if not isinstance(token, str) or not hmac.compare_digest(
                token, owner.token
            ):
                raise PermissionError("Worker authentication failed")
            action = request.get("action")
            payload = request.get("payload", {})
            if not isinstance(action, str) or not isinstance(payload, dict):
                raise ValueError("Invalid worker request fields")
            result = owner.dispatch(action, payload)
            owner._respond(self.wfile, True, result=result)
        except Exception as exc:
            owner._respond(self.wfile, False, error=str(exc))


class _UnixServer(socketserver.UnixStreamServer):
    allow_reuse_address = False


class WorkerServer:
    def __init__(
        self,
        manager: Any,
        paths: RuntimePaths | None = None,
        *,
        operations: dict[str, Operation] | None = None,
        service: Any | None = None,
        poll_interval: float = 0.5,
        read_timeout: float = 1.0,
        config_identity: str | None = None,
    ):
        self.manager = manager
        self.paths = paths or RuntimePaths.defaults()
        self.operations = operations or {}
        self.service = service
        self.poll_interval = poll_interval
        self.read_timeout = read_timeout
        self.config_identity = config_identity
        self.token = ""
        self._stopping = False
        self._server: _UnixServer | None = None
        self._runtime_lock_fd: int | None = None
        self._owns_runtime = False

    def _acquire_runtime_lock(self) -> None:
        root = self.paths.socket.parent
        if not root.is_absolute():
            raise RuntimeError("Worker runtime directory must be absolute")
        if root.is_symlink():
            raise RuntimeError("Worker runtime directory must not be a symlink")
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        info = root.stat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
            raise RuntimeError(
                "Worker runtime directory must be owned by the current user"
            )
        if stat.S_IMODE(info.st_mode) & 0o077:
            raise RuntimeError(
                "Worker runtime directory must not be group/world accessible"
            )
        lock_path = root / "worker.lock"
        flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(lock_path, flags, 0o600)
        try:
            os.fchmod(descriptor, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            os.close(descriptor)
            raise RuntimeError("Worker is already starting or running") from exc
        self._runtime_lock_fd = descriptor

    def _release_runtime_lock(self) -> None:
        descriptor = self._runtime_lock_fd
        if descriptor is None:
            return
        self._runtime_lock_fd = None
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)

    @staticmethod
    def _secure_write(path: Path, text: str) -> None:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())

    def _prepare(self) -> None:
        if self._runtime_lock_fd is None:
            raise RuntimeError("Worker runtime lock must be held before preparation")
        root = self.paths.socket.parent
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(root, 0o700)
        for path in (self.paths.socket, self.paths.pid, self.paths.token):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        self._owns_runtime = True
        self.token = secrets.token_hex(32)
        self._secure_write(self.paths.token, self.token)
        self._secure_write(self.paths.pid, str(os.getpid()))

    @staticmethod
    def _respond(stream, ok: bool, *, result: Any = None, error: str | None = None):
        payload = {"ok": ok, "result": result} if ok else {"ok": False, "error": error}
        encoded = (
            json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            + b"\n"
        )
        if len(encoded) > MAX_MESSAGE_BYTES:
            encoded = b'{"ok":false,"error":"Response exceeds the size limit"}\n'
        try:
            stream.write(encoded)
            stream.flush()
        except OSError:
            return

    def dispatch(self, action: str, payload: dict[str, Any]) -> Any:
        if action == "status":
            return {
                "pid": os.getpid(),
                "config_fingerprint": self.config_identity,
                "models": self.manager.status(),
                "index": self.service.status() if self.service is not None else None,
            }
        if action == "ping":
            models = payload.get("models", ["embedding", "reranker"])
            return {"kept": self.manager.ping(models)}
        if action == "unload":
            models = payload.get("models", ["embedding", "reranker"])
            return {"unloaded": self.manager.unload(models)}
        if action == "warm":
            models = payload.get("models", ["embedding", "reranker"])
            loaded = []
            request_context = getattr(self.manager, "request", None)
            context = request_context() if request_context else contextlib.nullcontext()
            with context:
                for name in models:
                    if name == "embedding":
                        self.manager.get_embedder()
                    elif name == "reranker":
                        self.manager.get_reranker(allow_disabled=True)
                    else:
                        raise ValueError(f"Unknown model name: {name}")
                    loaded.append(name)
            return {"loaded": loaded, "models": self.manager.status()}
        if action == "stop":
            self._stopping = True
            return {"stopping": True}
        operation = self.operations.get(action)
        if operation is None:
            raise ValueError(f"Unknown action: {action}")
        self.manager.tick(force_pressure=True)
        request_context = getattr(self.manager, "request", None)
        context = request_context() if request_context else contextlib.nullcontext()
        with context:
            return operation(payload)

    def _maintenance(self) -> None:
        model_evicted = self.manager.tick()
        consume_evictions = getattr(self.manager, "consume_evictions", None)
        if consume_evictions is not None:
            model_evicted = list(dict.fromkeys([*model_evicted, *consume_evictions()]))
        index_evicted = self.service.tick() if self.service is not None else []
        if not model_evicted and not index_evicted:
            return
        models = self.manager.status()
        index = self.service.status() if self.service is not None else None
        if (
            models.get("embedding") is None
            and models.get("reranker") is None
            and index is None
        ):
            self._stopping = True

    def _close_resources(self) -> None:
        if self.service is not None:
            self.service.close()
        self.manager.close()

    def serve(self) -> None:
        self._acquire_runtime_lock()
        try:
            self._prepare()
            server = _UnixServer(str(self.paths.socket), _RequestHandler)
            self._server = server
            server.owner = self  # type: ignore[attr-defined]
            server.timeout = self.poll_interval
            os.chmod(self.paths.socket, 0o600)
            while not self._stopping:
                server.handle_request()
                self._maintenance()
        finally:
            if self._server is not None:
                self._server.server_close()
            if self._owns_runtime:
                self._close_resources()
                for path in (self.paths.socket, self.paths.pid, self.paths.token):
                    with contextlib.suppress(FileNotFoundError):
                        path.unlink()
                self._owns_runtime = False
            self._release_runtime_lock()


def create_default_server(config_path: str | None = None) -> WorkerServer:
    config = load_config(config_path)

    def make_embedder(device: str):
        return QwenEmbedder(
            device=device,
            dtype="bf16",
            batch_size=4,
            local_files_only=False,
            attention_backend=config.attention_backend,
        )

    def make_reranker(device: str):
        return QwenReranker(
            device=device,
            dtype="bf16",
            local_files_only=False,
            attention_backend=config.attention_backend,
        )

    manager = ModelManager(
        config,
        embedder_factory=make_embedder,
        reranker_factory=make_reranker,
    )
    service = ModelService(
        manager,
        idle_timeout_seconds=config.idle_timeout_seconds,
    )
    return WorkerServer(
        manager,
        operations={"index": service.index, "search": service.search},
        service=service,
        poll_interval=min(0.5, config.pressure_poll_seconds),
        config_identity=config_fingerprint(config),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m graphify_embeddings.worker")
    parser.add_argument("--config")
    args = parser.parse_args(argv)
    server = create_default_server(args.config)

    def stop_server(_signum, _frame):
        server._stopping = True

    signal.signal(signal.SIGTERM, stop_server)
    signal.signal(signal.SIGINT, stop_server)
    server.serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
