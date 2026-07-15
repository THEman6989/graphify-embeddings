from __future__ import annotations

import fcntl
import json
import os
import socket
import stat
import struct
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import config_fingerprint, default_runtime_dir, load_config


MAX_MESSAGE_BYTES = 1024 * 1024


class WorkerError(RuntimeError):
    pass


class WorkerConfigMismatch(WorkerError):
    pass


def _peer_uid(connection: socket.socket) -> int:
    if not hasattr(socket, "SO_PEERCRED"):
        raise RuntimeError("SO_PEERCRED is required for worker authentication")
    credentials = connection.getsockopt(
        socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")
    )
    _pid, uid, _gid = struct.unpack("3i", credentials)
    return uid


@dataclass(frozen=True)
class RuntimePaths:
    socket: Path
    pid: Path
    token: Path

    @classmethod
    def defaults(cls) -> "RuntimePaths":
        root = default_runtime_dir()
        return cls(root / "worker.sock", root / "worker.pid", root / "worker.token")


class WorkerClient:
    def __init__(self, paths: RuntimePaths | None = None, *, timeout: float = 3600.0):
        self.paths = paths or RuntimePaths.defaults()
        self.timeout = timeout

    def request(self, action: str, payload: dict[str, Any] | None = None) -> Any:
        try:
            token = self.paths.token.read_text(encoding="ascii").strip()
        except OSError as exc:
            raise WorkerError("Worker authentication token is unavailable") from exc
        message = {"action": action, "token": token, "payload": payload or {}}
        encoded = (
            json.dumps(message, ensure_ascii=False, allow_nan=False).encode("utf-8")
            + b"\n"
        )
        if len(encoded) > MAX_MESSAGE_BYTES:
            raise WorkerError("Worker request exceeds the size limit")
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(self.timeout)
                connection.connect(str(self.paths.socket))
                if _peer_uid(connection) != os.getuid():
                    raise PermissionError("Worker server UID does not match the client")
                connection.sendall(encoded)
                response = bytearray()
                while not response.endswith(b"\n"):
                    chunk = connection.recv(65536)
                    if not chunk:
                        break
                    response.extend(chunk)
                    if len(response) > MAX_MESSAGE_BYTES:
                        raise WorkerError("Worker response exceeds the size limit")
        except (OSError, TimeoutError) as exc:
            raise WorkerError(f"Worker is unavailable: {exc}") from exc
        try:
            decoded = json.loads(bytes(response))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise WorkerError("Worker returned an invalid response") from exc
        if not isinstance(decoded, dict):
            raise WorkerError("Worker returned a non-object response")
        if not decoded.get("ok"):
            raise WorkerError(str(decoded.get("error", "Worker request failed")))
        return decoded.get("result")


def require_worker_config(
    client: WorkerClient, config_path: str | None = None
) -> dict[str, Any]:
    expected = config_fingerprint(load_config(config_path))
    status = client.request("status")
    if not isinstance(status, dict):
        raise WorkerError("Worker returned an invalid status response")
    actual = status.get("config_fingerprint")
    if actual != expected:
        raise WorkerConfigMismatch(
            "Running worker configuration does not match the requested configuration; "
            "stop the worker before switching configs"
        )
    return status


@contextmanager
def _startup_lock(paths: RuntimePaths):
    root = paths.socket.parent
    if not root.is_absolute():
        raise WorkerError("Worker runtime directory must be absolute")
    if root.is_symlink():
        raise WorkerError("Worker runtime directory must not be a symlink")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = root.stat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise WorkerError("Worker runtime directory must be owned by the current user")
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise WorkerError("Worker runtime directory must not be group/world accessible")
    lock_path = root / "startup.lock"
    flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def ensure_worker(
    *,
    config_path: str | None = None,
    paths: RuntimePaths | None = None,
    startup_timeout: float = 15.0,
) -> WorkerClient:
    paths = paths or RuntimePaths.defaults()
    client = WorkerClient(paths)
    try:
        require_worker_config(client, config_path)
        return client
    except WorkerConfigMismatch:
        raise
    except WorkerError:
        pass
    with _startup_lock(paths):
        try:
            require_worker_config(client, config_path)
            return client
        except WorkerConfigMismatch:
            raise
        except WorkerError:
            pass
        command = [sys.executable, "-I", "-m", "graphify_embeddings.worker"]
        if config_path:
            command.extend(["--config", str(Path(config_path).expanduser().resolve())])
        log_path = paths.socket.parent / "worker.log"
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.upper().startswith("PYTHON")
        }
        with log_path.open("ab") as log:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                start_new_session=True,
                close_fds=True,
                cwd=str(Path(sys.prefix).resolve()),
                env=environment,
            )
        deadline = time.monotonic() + startup_timeout
        child_exit_seen_at: float | None = None
        last_error = "worker did not become ready"
        while time.monotonic() < deadline:
            try:
                require_worker_config(client, config_path)
                return client
            except WorkerConfigMismatch:
                raise
            except WorkerError as exc:
                last_error = str(exc)
            returncode = process.poll()
            if returncode is not None:
                now = time.monotonic()
                child_exit_seen_at = child_exit_seen_at or now
                last_error = f"worker exited with code {returncode}"
                if now - child_exit_seen_at >= 1.0:
                    break
            time.sleep(0.05)
        raise WorkerError(f"Could not start worker: {last_error}; log: {log_path}")
