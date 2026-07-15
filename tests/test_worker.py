from __future__ import annotations

import contextlib
import io
import tempfile
import threading
import time
import unittest
import json
import os
import subprocess
import struct
import sys
from pathlib import Path

import numpy as np

from graphify_embeddings.config import WorkerConfig, load_config
from graphify_embeddings import cli
from graphify_embeddings import worker as worker_module
from graphify_embeddings.gpu import (
    GpuInfo,
    GpuProcess,
    plan_model_devices,
    pressured_gpus,
)
from graphify_embeddings.manager import ModelManager
from graphify_embeddings.client import (
    WorkerClient,
    WorkerError,
    _peer_uid as _client_peer_uid,
    _startup_lock,
    ensure_worker,
)
from graphify_embeddings.worker import RuntimePaths, WorkerServer, _peer_uid
from graphify_embeddings.service import ModelService
from graphify_embeddings.models import _empty_device_cache, resolve_attention_backend


class FakeModel:
    def __init__(self, device: str):
        self.device = device
        self.closed = False

    def close(self):
        self.closed = True


class WorkerCoreTests(unittest.TestCase):
    def test_installer_no_flash_is_locked_noneditable_and_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            log = root / "uv.log"
            fake_uv = fake_bin / "uv"
            fake_uv.write_text(
                """#!/usr/bin/env python3
import os
from pathlib import Path
import sys
with Path(os.environ["FAKE_UV_LOG"]).open("a", encoding="utf-8") as handle:
    handle.write(" ".join(sys.argv[1:]) + "\\n")
if sys.argv[1] == "venv":
    venv = Path(sys.argv[-1])
    (venv / "bin").mkdir(parents=True, exist_ok=True)
    (venv / "bin" / "python").write_text(
        "#!/bin/sh\\nif [ \\"$1\\" = \\"-\\" ]; then cat >/dev/null; printf '{}\\n'; fi\\nexit 0\\n"
    )
    (venv / "bin" / "graphify-embeddings").write_text("#!/bin/sh\\nexit 0\\n")
    (venv / "bin" / "python").chmod(0o755)
    (venv / "bin" / "graphify-embeddings").chmod(0o755)
""",
                encoding="utf-8",
            )
            fake_uv.chmod(0o755)
            environment = {
                **os.environ,
                "PATH": f"{fake_bin}:{os.environ['PATH']}",
                "HOME": str(root / "home"),
                "GRAPHIFY_EMBEDDINGS_VENV": str(root / "venv"),
                "FAKE_UV_LOG": str(log),
            }
            installer = Path(__file__).parents[1] / "install.sh"
            for _ in range(2):
                subprocess.run(
                    [str(installer), "--no-flash-attn"],
                    check=True,
                    env=environment,
                    text=True,
                    capture_output=True,
                )

            commands = log.read_text(encoding="utf-8").splitlines()
            syncs = [command for command in commands if command.startswith("sync ")]
            uninstalls = [
                command
                for command in commands
                if command.startswith("pip uninstall ")
                and command.endswith(" flash-attn")
            ]
            self.assertEqual(len(syncs), 2)
            for command in syncs:
                self.assertIn("--locked", command)
                self.assertIn("--no-editable", command)
                self.assertIn("--active", command)
                self.assertIn("--extra gpu", command)
            self.assertEqual(len(uninstalls), 2)

    def test_linux_peer_credentials_extract_the_calling_uid(self):
        class FakeConnection:
            def getsockopt(self, _level, _option, _size):
                return struct.pack("3i", 1234, 5678, 9012)

        self.assertEqual(_peer_uid(FakeConnection()), 5678)
        self.assertEqual(_client_peer_uid(FakeConnection()), 5678)

    def test_startup_lock_rejects_symlinked_runtime_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real = root / "real"
            real.mkdir(mode=0o700)
            link = root / "link"
            link.symlink_to(real, target_is_directory=True)
            paths = RuntimePaths(
                link / "worker.sock", link / "worker.pid", link / "worker.token"
            )
            with self.assertRaisesRegex(WorkerError, "must not be a symlink"):
                with _startup_lock(paths):
                    pass

    def test_cuda_cache_cleanup_uses_the_models_device_context(self):
        events = []

        class DeviceContext:
            def __init__(self, device):
                self.device = device

            def __enter__(self):
                events.append(("enter", self.device))

            def __exit__(self, *_args):
                events.append(("exit", self.device))

        class FakeCuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def device(device):
                return DeviceContext(device)

            @staticmethod
            def empty_cache():
                events.append(("empty", None))

        class FakeTorch:
            cuda = FakeCuda()

        _empty_device_cache(FakeTorch(), "cuda:1")
        self.assertEqual(
            events,
            [("enter", "cuda:1"), ("empty", None), ("exit", "cuda:1")],
        )

    def test_attention_backend_auto_uses_flash_only_when_installed(self):
        from unittest.mock import patch

        with patch("importlib.util.find_spec", return_value=object()):
            self.assertEqual(resolve_attention_backend("auto"), "flash_attention_2")
        with patch("importlib.util.find_spec", return_value=None):
            self.assertEqual(resolve_attention_backend("auto"), "sdpa")
        self.assertEqual(resolve_attention_backend("sdpa"), "sdpa")
        with self.assertRaisesRegex(ValueError, "attention backend"):
            resolve_attention_backend("invalid")

    def test_cli_exposes_boolean_reranker_worker_and_pipeline_commands(self):
        parser = cli.build_parser()
        default_search = parser.parse_args(["search", "query"])
        self.assertIsNone(default_search.rerank)
        self.assertTrue(parser.parse_args(["search", "query", "--rerank"]).rerank)
        self.assertFalse(parser.parse_args(["search", "query", "--no-rerank"]).rerank)
        ping = parser.parse_args(["worker", "ping", "--models", "embedding"])
        self.assertEqual(ping.worker_action, "ping")
        self.assertEqual(ping.models, ["embedding"])
        pipeline = parser.parse_args(["pipeline", "/tmp/project", "--no-rerank"])
        self.assertEqual(pipeline.path, "/tmp/project")
        self.assertFalse(pipeline.rerank)

    def test_worker_rpc_resolves_relative_graph_before_cwd_changes(self):
        from unittest.mock import patch

        requests = []

        class FakeClient:
            def request(self, action, payload):
                requests.append((action, payload))
                if action == "index":
                    return {"nodes": 1}
                return {"index_build": None, "results": []}

        with tempfile.TemporaryDirectory() as temporary:
            previous = Path.cwd()
            os.chdir(temporary)
            try:
                parser = cli.build_parser()
                index_args = parser.parse_args(
                    ["index", "--graph", "graphify-out/graph.json"]
                )
                search_args = parser.parse_args(
                    ["search", "query", "--graph", "graphify-out/graph.json", "--json"]
                )
                with (
                    patch(
                        "graphify_embeddings.cli.load_config",
                        return_value=WorkerConfig(),
                    ),
                    patch(
                        "graphify_embeddings.cli._worker_client",
                        return_value=FakeClient(),
                    ),
                    patch("builtins.print"),
                ):
                    cli.command_index(index_args)
                    cli.command_search(search_args)
            finally:
                os.chdir(previous)

        expected = str(Path(temporary) / "graphify-out" / "graph.json")
        self.assertEqual(
            [payload["graph"] for _action, payload in requests], [expected, expected]
        )

    def test_pipeline_one_shot_emits_one_json_document(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            graph_path = project / "graphify-out" / "graph.json"
            graph_path.parent.mkdir()
            graph_path.write_text("{}", encoding="utf-8")
            args = cli.build_parser().parse_args(
                ["pipeline", str(project), "--no-link"]
            )
            output = io.StringIO()
            with (
                patch(
                    "graphify_embeddings.cli.subprocess.run",
                    return_value=subprocess.CompletedProcess([], 0, "", ""),
                ),
                patch(
                    "graphify_embeddings.cli.load_config",
                    return_value=WorkerConfig(
                        worker_enabled=False, attention_backend="sdpa"
                    ),
                ),
                patch(
                    "graphify_embeddings.cli._index_stats",
                    return_value={"nodes": 1},
                ) as index_stats,
                contextlib.redirect_stdout(output),
            ):
                cli.command_pipeline(args)

        payload = json.loads(output.getvalue())
        self.assertEqual(payload["index"], {"nodes": 1, "worker": False})
        self.assertEqual(index_stats.call_args.args[0].attention_backend, "sdpa")

    def test_default_cli_attention_defers_to_the_worker_config(self):
        from unittest.mock import patch

        parser = cli.build_parser()
        args = parser.parse_args(["index"])
        self.assertTrue(
            cli._worker_compatible(args, WorkerConfig(attention_backend="sdpa"))
        )
        offline_args = parser.parse_args(["index", "--local-files-only"])
        self.assertFalse(cli._worker_compatible(offline_args, WorkerConfig()))

        configured = WorkerConfig(worker_enabled=False, attention_backend="sdpa")
        seen_backends = []

        class StopOneShot(RuntimeError):
            pass

        def stop_embedder(args):
            seen_backends.append(args.attention_backend)
            raise StopOneShot

        index_args = parser.parse_args(["index", "--no-worker"])
        with (
            patch("graphify_embeddings.cli.load_config", return_value=configured),
            patch("graphify_embeddings.cli.GraphifyGraph"),
            patch("graphify_embeddings.cli.EmbeddingIndex"),
            patch(
                "graphify_embeddings.cli._embedder_from_args", side_effect=stop_embedder
            ),
            self.assertRaises(StopOneShot),
        ):
            cli._index_stats(index_args)

        search_args = parser.parse_args(["search", "query", "--no-worker"])

        def stop_index(_graph, args, *, keep_embedder):
            self.assertTrue(keep_embedder)
            seen_backends.append(args.attention_backend)
            raise StopOneShot

        with (
            patch("graphify_embeddings.cli.load_config", return_value=configured),
            patch("graphify_embeddings.cli.GraphifyGraph"),
            patch("graphify_embeddings.cli._ensure_index", side_effect=stop_index),
            self.assertRaises(StopOneShot),
        ):
            cli.command_search(search_args)

        self.assertEqual(seen_backends, ["sdpa", "sdpa"])

    def test_auto_start_false_uses_running_worker_without_spawning(self):
        from unittest.mock import patch

        args = cli.build_parser().parse_args(["index"])
        config = WorkerConfig(auto_start=False)
        with (
            patch.object(WorkerClient, "request", return_value={"pid": 123}),
            patch("graphify_embeddings.cli.ensure_worker") as spawn,
        ):
            client = cli._worker_client(args, config)
        self.assertIsInstance(client, WorkerClient)
        spawn.assert_not_called()

        with (
            patch.object(WorkerClient, "request", side_effect=WorkerError("offline")),
            patch("graphify_embeddings.cli.ensure_worker") as spawn,
        ):
            self.assertIsNone(cli._worker_client(args, config))
        spawn.assert_not_called()

    def test_default_worker_allows_pinned_model_snapshot_downloads(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "missing.toml"
            with (
                patch("graphify_embeddings.worker.QwenEmbedder") as embedder,
                patch("graphify_embeddings.worker.QwenReranker") as reranker,
            ):
                server = worker_module.create_default_server(str(config_path))
                server.manager._embedder_factory("cuda:1")
                server.manager._reranker_factory("cuda:0")
        self.assertFalse(embedder.call_args.kwargs["local_files_only"])
        self.assertFalse(reranker.call_args.kwargs["local_files_only"])

    def test_config_defaults_and_toml_overrides(self):
        defaults = load_config(path=None)
        self.assertTrue(defaults.worker_enabled)
        self.assertEqual(defaults.idle_timeout_seconds, 60.0)
        self.assertTrue(defaults.reranker_enabled)
        self.assertEqual(defaults.placement_policy, "split")
        self.assertEqual(defaults.preferred_gpu, 0)
        self.assertEqual(defaults.secondary_gpu, 1)
        self.assertEqual(defaults.min_free_gib, 6.0)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.toml"
            path.write_text(
                """
[worker]
enabled = false
idle_timeout_seconds = 90
reranker_enabled = false

[gpu]
placement_policy = "sequential"
preferred_gpu = 1
secondary_gpu = 0
min_free_gib = 4.5
""",
                encoding="utf-8",
            )
            configured = load_config(path=path)
        self.assertFalse(configured.worker_enabled)
        self.assertEqual(configured.idle_timeout_seconds, 90.0)
        self.assertFalse(configured.reranker_enabled)
        self.assertEqual(configured.placement_policy, "sequential")
        self.assertEqual(configured.preferred_gpu, 1)
        self.assertEqual(configured.min_free_gib, 4.5)

    def test_invalid_config_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "idle_timeout_seconds"):
            WorkerConfig(idle_timeout_seconds=0).validate()
        with self.assertRaisesRegex(ValueError, "placement_policy"):
            WorkerConfig(placement_policy="random").validate()
        with self.assertRaisesRegex(ValueError, "embedding_device"):
            WorkerConfig(embedding_device="cuda:-1").validate()
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(idle_timeout_seconds=value):
                with self.assertRaisesRegex(ValueError, "idle_timeout_seconds"):
                    WorkerConfig(idle_timeout_seconds=value).validate()
            with self.subTest(pressure_poll_seconds=value):
                with self.assertRaisesRegex(ValueError, "pressure_poll_seconds"):
                    WorkerConfig(pressure_poll_seconds=value).validate()
            invalid_gib_configs = (
                ("min_free_gib", WorkerConfig(min_free_gib=value)),
                (
                    "embedding_required_gib",
                    WorkerConfig(embedding_required_gib=value),
                ),
                (
                    "reranker_required_gib",
                    WorkerConfig(reranker_required_gib=value),
                ),
            )
            for field, invalid_config in invalid_gib_configs:
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(ValueError, "GiB"):
                        invalid_config.validate()

    def test_split_placement_uses_5090_for_reranker_and_3090_for_embedder(self):
        gpus = [
            GpuInfo(0, "NVIDIA GeForce RTX 5090", 31.396, 29.846, ()),
            GpuInfo(1, "NVIDIA GeForce RTX 3090", 24.0, 23.5, ()),
        ]
        plan = plan_model_devices(gpus, WorkerConfig())
        self.assertEqual(plan.embedding_device, "cuda:1")
        self.assertEqual(plan.reranker_device, "cuda:0")
        self.assertFalse(plan.same_gpu)

    def test_models_are_not_unsafe_co_residents_on_5090(self):
        gpus = [GpuInfo(0, "NVIDIA GeForce RTX 5090", 31.396, 29.846, ())]
        plan = plan_model_devices(gpus, WorkerConfig(secondary_gpu=0))
        self.assertEqual(plan.reranker_device, "cuda:0")
        self.assertIsNone(plan.embedding_device)
        self.assertFalse(plan.same_gpu)

    def test_external_pressure_has_priority_over_residency(self):
        gpus = [
            GpuInfo(
                0,
                "NVIDIA GeForce RTX 5090",
                31.396,
                12.0,
                (GpuProcess(99, "ComfyUI", 4096),),
            ),
            GpuInfo(1, "NVIDIA GeForce RTX 3090", 24.0, 3.0, ()),
        ]
        pressured = pressured_gpus(gpus, WorkerConfig(), own_pid=123)
        self.assertEqual(pressured, {0, 1})

    def test_manager_reuses_models_ping_extends_lease_and_idle_unloads(self):
        now = [100.0]
        created: list[FakeModel] = []

        def factory(device: str):
            model = FakeModel(device)
            created.append(model)
            return model

        def inventory():
            return [
                GpuInfo(0, "5090", 31.396, 29.0, ()),
                GpuInfo(1, "3090", 24.0, 23.0, ()),
            ]

        manager = ModelManager(
            WorkerConfig(idle_timeout_seconds=60),
            embedder_factory=factory,
            reranker_factory=factory,
            inventory_provider=inventory,
            clock=lambda: now[0],
            own_pid=123,
        )
        embedder = manager.get_embedder()
        self.assertIs(embedder, manager.get_embedder())
        self.assertEqual(embedder.device, "cuda:1")
        now[0] = 150.0
        self.assertEqual(manager.ping(["embedding"]), ["embedding"])
        now[0] = 205.0
        manager.tick()
        self.assertFalse(embedder.closed)
        now[0] = 211.0
        manager.tick()
        self.assertTrue(embedder.closed)
        self.assertIsNone(manager.status()["embedding"])

    def test_lease_starts_after_a_long_request_finishes(self):
        now = [0.0]
        manager = ModelManager(
            WorkerConfig(idle_timeout_seconds=60),
            embedder_factory=FakeModel,
            reranker_factory=FakeModel,
            inventory_provider=lambda: [
                GpuInfo(0, "5090", 31.396, 29.0, ()),
                GpuInfo(1, "3090", 24.0, 23.0, ()),
            ],
            clock=lambda: now[0],
            own_pid=123,
        )
        with manager.request():
            embedder = manager.get_embedder()
            now[0] = 65.0
            self.assertEqual(manager.tick(), [])
        now[0] = 124.0
        manager.tick()
        self.assertFalse(embedder.closed)
        now[0] = 126.0
        manager.tick()
        self.assertTrue(embedder.closed)

    def test_request_renews_only_models_used_by_that_request(self):
        now = [0.0]
        manager = ModelManager(
            WorkerConfig(idle_timeout_seconds=60),
            embedder_factory=FakeModel,
            reranker_factory=FakeModel,
            inventory_provider=lambda: [
                GpuInfo(0, "5090", 31.396, 29.0, ()),
                GpuInfo(1, "3090", 24.0, 23.0, ()),
            ],
            clock=lambda: now[0],
            own_pid=123,
        )
        embedder = manager.get_embedder()
        reranker = manager.get_reranker()
        now[0] = 50.0
        with manager.request():
            self.assertIs(manager.get_embedder(), embedder)
            now[0] = 70.0

        self.assertEqual(manager.tick(), ["reranker"])
        self.assertFalse(embedder.closed)
        self.assertTrue(reranker.closed)

    def test_warm_lease_starts_after_all_requested_models_load(self):
        now = [0.0]

        def reranker_factory(device):
            now[0] = 65.0
            return FakeModel(device)

        manager = ModelManager(
            WorkerConfig(idle_timeout_seconds=60),
            embedder_factory=FakeModel,
            reranker_factory=reranker_factory,
            inventory_provider=lambda: [
                GpuInfo(0, "5090", 31.396, 29.0, ()),
                GpuInfo(1, "3090", 24.0, 23.0, ()),
            ],
            clock=lambda: now[0],
            own_pid=123,
        )
        server = WorkerServer(manager)
        server.dispatch("warm", {"models": ["embedding", "reranker"]})
        manager.tick()
        status = manager.status()
        self.assertIsNotNone(status["embedding"])
        self.assertIsNotNone(status["reranker"])

    def test_manager_can_disable_reranker_and_evict_under_pressure(self):
        now = [0.0]
        current_inventory = [
            GpuInfo(0, "5090", 31.396, 29.0, ()),
            GpuInfo(1, "3090", 24.0, 23.0, ()),
        ]
        created: list[FakeModel] = []

        def factory(device: str):
            model = FakeModel(device)
            created.append(model)
            return model

        disabled = ModelManager(
            WorkerConfig(reranker_enabled=False),
            embedder_factory=factory,
            reranker_factory=factory,
            inventory_provider=lambda: current_inventory,
            clock=lambda: now[0],
            own_pid=123,
        )
        with self.assertRaisesRegex(RuntimeError, "disabled"):
            disabled.get_reranker()
        warm_result = WorkerServer(disabled).dispatch("warm", {"models": ["reranker"]})
        self.assertEqual(warm_result["loaded"], ["reranker"])
        self.assertIsNotNone(disabled.status()["reranker"])

        manager = ModelManager(
            WorkerConfig(),
            embedder_factory=factory,
            reranker_factory=factory,
            inventory_provider=lambda: current_inventory,
            clock=lambda: now[0],
            own_pid=123,
        )
        embedding = manager.get_embedder()
        reranker = manager.get_reranker()
        current_inventory[:] = [
            GpuInfo(0, "5090", 31.396, 10.0, (GpuProcess(77, "game", 6000),)),
            GpuInfo(1, "3090", 24.0, 2.0, ()),
        ]
        now[0] = 3.0
        manager.tick()
        self.assertTrue(embedding.closed)
        self.assertTrue(reranker.closed)
        self.assertIsNone(manager.status()["embedding"])
        self.assertIsNone(manager.status()["reranker"])

    def test_manager_migrates_requested_model_when_preferred_gpu_is_busy(self):
        pressure = [False]
        created: list[FakeModel] = []

        def factory(device: str):
            model = FakeModel(device)
            created.append(model)
            return model

        def inventory():
            embedding = created[0] if created else None
            gpu1_free = 23.0 if embedding is None or embedding.closed else 7.5
            gpu0_processes = (GpuProcess(88, "ComfyUI", 8192),) if pressure[0] else ()
            return [
                GpuInfo(0, "5090", 31.396, 20.0, gpu0_processes),
                GpuInfo(1, "3090", 24.0, gpu1_free, ()),
            ]

        manager = ModelManager(
            WorkerConfig(),
            embedder_factory=factory,
            reranker_factory=factory,
            inventory_provider=inventory,
            own_pid=123,
        )
        embedding = manager.get_embedder()
        self.assertEqual(embedding.device, "cuda:1")
        pressure[0] = True
        reranker = manager.get_reranker()
        self.assertTrue(embedding.closed)
        self.assertEqual(reranker.device, "cuda:1")


class FakeManager:
    def __init__(self):
        self.closed = False
        self.ticks = 0
        self.loaded = {"embedding", "reranker"}

    def status(self):
        return {
            "embedding": {"device": "cuda:1"} if "embedding" in self.loaded else None,
            "reranker": {"device": "cuda:0"} if "reranker" in self.loaded else None,
        }

    def ping(self, models):
        return [name for name in models if name in self.loaded]

    def unload(self, models):
        removed = []
        for name in models:
            if name in self.loaded:
                self.loaded.remove(name)
                removed.append(name)
        return removed

    def tick(self, *, force_pressure=False):
        self.ticks += 1
        return []

    def close(self):
        self.closed = True
        self.loaded.clear()


class FakeIndexService:
    def __init__(self):
        self.ticks = 0
        self.closed = False

    def status(self):
        return {"graph": "/tmp/graph.json", "node_count": 48501, "dimension": 4096}

    def tick(self):
        self.ticks += 1
        return []

    def close(self):
        self.closed = True


class WorkerIpcTests(unittest.TestCase):
    def test_worker_maintenance_status_and_shutdown_manage_resident_index(self):
        service = FakeIndexService()
        server = WorkerServer(FakeManager(), service=service)
        status = server.dispatch("status", {})
        self.assertEqual(status["index"]["node_count"], 48501)
        server._maintenance()
        self.assertEqual(service.ticks, 1)
        server._close_resources()
        self.assertTrue(service.closed)

    def test_status_invalidation_of_last_index_stops_empty_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            graph_dir = Path(temporary) / "graphify-out"
            graph_dir.mkdir()
            graph_path = graph_dir / "graph.json"
            graph_path.write_text(
                json.dumps(
                    {
                        "directed": False,
                        "multigraph": False,
                        "graph": {},
                        "nodes": [{"id": "a", "label": "cache", "docstring": "cache"}],
                        "links": [],
                        "hyperedges": [],
                    }
                ),
                encoding="utf-8",
            )
            service = ModelService(ServiceManager())
            service.search({"graph": str(graph_path), "query": "cache", "neighbors": 0})
            manager = FakeManager()
            manager.loaded.clear()
            server = WorkerServer(manager, service=service)
            metadata_path = graph_dir / "cache" / "embeddings.json"
            metadata_path.write_text(
                metadata_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
            )

            self.assertIsNone(server.dispatch("status", {})["index"])
            server._maintenance()
            self.assertTrue(server._stopping)

    def test_timeout_seen_through_status_remains_pending_for_maintenance(self):
        now = [10.0]
        with tempfile.TemporaryDirectory() as temporary:
            graph_dir = Path(temporary) / "graphify-out"
            graph_dir.mkdir()
            graph_path = graph_dir / "graph.json"
            graph_path.write_text(
                json.dumps(
                    {
                        "directed": False,
                        "multigraph": False,
                        "graph": {},
                        "nodes": [{"id": "a", "label": "cache", "docstring": "cache"}],
                        "links": [],
                        "hyperedges": [],
                    }
                ),
                encoding="utf-8",
            )
            service = ModelService(
                ServiceManager(), idle_timeout_seconds=60, clock=lambda: now[0]
            )
            service.search({"graph": str(graph_path), "query": "cache", "neighbors": 0})
            manager = FakeManager()
            manager.loaded.clear()
            server = WorkerServer(manager, service=service)
            now[0] = 70.0

            self.assertIsNone(server.dispatch("status", {})["index"])
            server._maintenance()
            self.assertTrue(server._stopping)

    def test_maintenance_keeps_never_used_empty_worker_running(self):
        manager = FakeManager()
        manager.loaded.clear()
        server = WorkerServer(manager, service=ModelService(ServiceManager()))
        server._maintenance()
        self.assertFalse(server._stopping)

    def test_worker_stops_when_index_timeout_leaves_no_resident_resources(self):
        class IndexTimeoutService(FakeIndexService):
            def tick(self):
                self.ticks += 1
                return ["index"]

            def status(self):
                return None

        manager = FakeManager()
        manager.loaded.clear()
        server = WorkerServer(manager, service=IndexTimeoutService())
        server._maintenance()
        self.assertTrue(server._stopping)

    def test_failed_reload_after_preflight_eviction_stops_empty_worker(self):
        pressured = [False]

        def inventory():
            free = 1.0 if pressured[0] else 23.0
            return [
                GpuInfo(0, "5090", 31.396, free, ()),
                GpuInfo(1, "3090", 24.0, free, ()),
            ]

        manager = ModelManager(
            WorkerConfig(),
            embedder_factory=FakeModel,
            reranker_factory=FakeModel,
            inventory_provider=inventory,
            own_pid=123,
        )
        embedder = manager.get_embedder()
        server = WorkerServer(
            manager,
            operations={"search": lambda _payload: manager.get_embedder()},
        )
        pressured[0] = True

        with self.assertRaisesRegex(RuntimeError, "No GPU"):
            server.dispatch("search", {})
        self.assertTrue(embedder.closed)
        server._maintenance()
        self.assertTrue(server._stopping)

    def test_worker_stops_after_eviction_leaves_no_resident_models(self):
        class EmptyAfterEvictionManager(FakeManager):
            def tick(self, *, force_pressure=False):
                self.ticks += 1
                return ["embedding"]

            def status(self):
                return {"embedding": None, "reranker": None}

        server = WorkerServer(EmptyAfterEvictionManager())
        server._maintenance()
        self.assertTrue(server._stopping)

    def test_runtime_lock_makes_live_stale_pid_safe_to_replace(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = RuntimePaths(
                root / "worker.sock", root / "worker.pid", root / "worker.token"
            )
            paths.socket.write_text("stale", encoding="ascii")
            paths.pid.write_text(str(os.getpid()), encoding="ascii")
            paths.token.write_text("stale", encoding="ascii")
            server = WorkerServer(FakeManager(), paths=paths)

            server._acquire_runtime_lock()
            try:
                server._prepare()
                self.assertFalse(paths.socket.exists())
                self.assertEqual(
                    paths.pid.read_text(encoding="ascii"), str(os.getpid())
                )
                self.assertNotEqual(paths.token.read_text(encoding="ascii"), "stale")
            finally:
                for path in (paths.socket, paths.pid, paths.token):
                    path.unlink(missing_ok=True)
                server._release_runtime_lock()

    def test_runtime_lock_rejects_second_server_before_prepare(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = RuntimePaths(
                root / "worker.sock", root / "worker.pid", root / "worker.token"
            )
            first = WorkerServer(FakeManager(), paths=paths)
            second = WorkerServer(FakeManager(), paths=paths)
            first._acquire_runtime_lock()
            try:
                with self.assertRaisesRegex(
                    RuntimeError, "already starting or running"
                ):
                    second._acquire_runtime_lock()
            finally:
                first._release_runtime_lock()

    def test_concurrent_autostart_spawns_exactly_one_worker(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = RuntimePaths(
                root / "worker.sock", root / "worker.pid", root / "worker.token"
            )
            barrier = threading.Barrier(2)
            state_lock = threading.Lock()
            calls: dict[int, int] = {}
            started = [False]
            spawn_count = [0]

            def request(_client, _action, _payload=None):
                ident = threading.get_ident()
                with state_lock:
                    calls[ident] = calls.get(ident, 0) + 1
                    call_number = calls[ident]
                    ready = started[0]
                if call_number == 1:
                    barrier.wait(timeout=2)
                    raise WorkerError("not running")
                if ready:
                    return {"pid": 123}
                raise WorkerError("not running")

            class FakeProcess:
                returncode = None

                def __init__(self, *_args, **_kwargs):
                    with state_lock:
                        spawn_count[0] += 1
                        started[0] = True

                @staticmethod
                def poll():
                    return None

            results = []
            errors = []

            def start():
                try:
                    results.append(ensure_worker(paths=paths, startup_timeout=1))
                except Exception as exc:
                    errors.append(exc)

            with (
                patch.object(WorkerClient, "request", request),
                patch("graphify_embeddings.client.subprocess.Popen", FakeProcess),
            ):
                threads = [threading.Thread(target=start) for _ in range(2)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=3)

            self.assertEqual(errors, [])
            self.assertEqual(len(results), 2)
            self.assertEqual(spawn_count[0], 1)

    def test_worker_autostart_uses_isolated_interpreter_and_trusted_context(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = RuntimePaths(
                root / "worker.sock", root / "worker.pid", root / "worker.token"
            )
            config = root / "config.toml"
            config.write_text("[worker]\n", encoding="utf-8")
            requests = [
                WorkerError("not running"),
                WorkerError("not running"),
                {"pid": 123},
            ]
            popen_calls = []

            def request(_client, _action, _payload=None):
                result = requests.pop(0)
                if isinstance(result, Exception):
                    raise result
                return result

            class FakeProcess:
                returncode = None

                def __init__(self, command, **kwargs):
                    popen_calls.append((command, kwargs))

                @staticmethod
                def poll():
                    return None

            with (
                patch.object(WorkerClient, "request", request),
                patch("graphify_embeddings.client.subprocess.Popen", FakeProcess),
                patch.dict(
                    "os.environ", {"PYTHONPATH": "/tmp/evil", "PYTHONHOME": "/tmp/evil"}
                ),
            ):
                ensure_worker(config_path=str(config), paths=paths, startup_timeout=1)

            self.assertEqual(len(popen_calls), 1)
            command, kwargs = popen_calls[0]
            self.assertEqual(
                command[:4], [sys.executable, "-I", "-m", "graphify_embeddings.worker"]
            )
            self.assertEqual(command[4:], ["--config", str(config.resolve())])
            self.assertEqual(Path(kwargs["cwd"]), Path(sys.prefix).resolve())
            self.assertNotIn("PYTHONPATH", kwargs["env"])
            self.assertNotIn("PYTHONHOME", kwargs["env"])

    def test_default_client_timeout_supports_large_model_loads(self):
        self.assertGreaterEqual(WorkerClient().timeout, 600.0)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.paths = RuntimePaths(
            socket=root / "worker.sock",
            pid=root / "worker.pid",
            token=root / "worker.token",
        )
        self.manager = FakeManager()
        self.server = WorkerServer(
            self.manager,
            self.paths,
            operations={"echo": lambda payload: {"value": payload["value"]}},
            poll_interval=0.01,
        )
        self.thread = threading.Thread(target=self.server.serve, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 2
        while not self.paths.socket.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.client = WorkerClient(self.paths, timeout=2)

    def tearDown(self):
        if self.thread.is_alive():
            try:
                self.client.request("stop")
            except Exception:
                pass
        self.thread.join(timeout=2)
        self.temporary.cleanup()

    def test_status_ping_unload_operation_and_clean_stop(self):
        self.assertEqual(self.paths.socket.stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.paths.token.stat().st_mode & 0o777, 0o600)
        status = self.client.request("status")
        self.assertEqual(status["models"]["reranker"]["device"], "cuda:0")
        ping = self.client.request("ping", {"models": ["embedding"]})
        self.assertEqual(ping["kept"], ["embedding"])
        echo = self.client.request("echo", {"value": "safe"})
        self.assertEqual(echo, {"value": "safe"})
        unloaded = self.client.request("unload", {"models": ["reranker"]})
        self.assertEqual(unloaded["unloaded"], ["reranker"])
        self.client.request("stop")
        self.thread.join(timeout=2)
        self.assertFalse(self.thread.is_alive())
        self.assertTrue(self.manager.closed)
        self.assertFalse(self.paths.socket.exists())
        self.assertFalse(self.paths.pid.exists())
        self.assertFalse(self.paths.token.exists())

    def test_authentication_and_unknown_actions_fail_closed(self):
        original = self.paths.token.read_text(encoding="ascii")
        self.paths.token.write_text("wrong-token", encoding="ascii")
        with self.assertRaisesRegex(WorkerError, "authentication"):
            self.client.request("status")
        self.paths.token.write_text(original, encoding="ascii")
        with self.assertRaisesRegex(WorkerError, "Unknown action"):
            self.client.request("not-an-action")


class ServiceEmbedder(FakeModel):
    model_name = "fake/embedder"
    backend = "fake"
    instruction = "test"
    requested_dtype = "fp32"

    def cache_identity(self):
        return {
            "model": self.model_name,
            "backend": self.backend,
            "instruction": self.instruction,
            "revision": "fixed",
            "wrapper_sha256": None,
            "artifact_fingerprint": None,
            "dtype": self.requested_dtype,
        }

    def encode(self, texts, *, show_progress=False):
        vectors = []
        for text in texts:
            vector = np.asarray(
                [text.lower().count("cache") + 0.1, text.lower().count("gpu") + 0.1],
                dtype=np.float32,
            )
            vector /= np.linalg.norm(vector)
            vectors.append(vector)
        return np.stack(vectors) if vectors else np.empty((0, 2), dtype=np.float32)


class ServiceReranker(FakeModel):
    model_name = "fake/reranker"

    def score(self, query, documents, batch_size=1):
        return np.asarray([1.0 if "gpu" in text.lower() else 0.0 for text in documents])


class ServiceManager:
    def __init__(self):
        self.embedding = ServiceEmbedder("cuda:1")
        self.reranker = ServiceReranker("cuda:0")
        self.reranker_overrides = []

    def get_embedder(self):
        return self.embedding

    def get_reranker(self, *, allow_disabled=False):
        self.reranker_overrides.append(allow_disabled)
        return self.reranker


class ModelServiceTests(unittest.TestCase):
    def test_second_search_reuses_resident_index(self):
        from unittest.mock import patch

        from graphify_embeddings.index import EmbeddingIndex

        with tempfile.TemporaryDirectory() as temporary:
            graph_dir = Path(temporary) / "graphify-out"
            graph_dir.mkdir()
            graph_path = graph_dir / "graph.json"
            graph_path.write_text(
                json.dumps(
                    {
                        "directed": False,
                        "multigraph": False,
                        "graph": {},
                        "nodes": [
                            {
                                "id": "a",
                                "label": "cache_loader",
                                "docstring": "cache vectors",
                            },
                            {
                                "id": "b",
                                "label": "gpu_runner",
                                "docstring": "gpu model",
                            },
                        ],
                        "links": [],
                        "hyperedges": [],
                    }
                ),
                encoding="utf-8",
            )
            service = ModelService(ServiceManager())
            service.index({"graph": str(graph_path), "force": False})
            real_load = EmbeddingIndex.load
            loads = []

            def tracked_load(index):
                loads.append(index.vectors_path)
                return real_load(index)

            with patch.object(EmbeddingIndex, "load", tracked_load):
                first = service.search(
                    {"graph": str(graph_path), "query": "cache", "neighbors": 0}
                )
                second = service.search(
                    {"graph": str(graph_path), "query": "gpu", "neighbors": 0}
                )

            self.assertEqual(first["results"][0]["id"], "a")
            self.assertEqual(second["results"][0]["id"], "b")
            self.assertEqual(len(loads), 1)

    def test_index_unloads_at_idle_timeout(self):
        now = [100.0]
        with tempfile.TemporaryDirectory() as temporary:
            graph_dir = Path(temporary) / "graphify-out"
            graph_dir.mkdir()
            graph_path = graph_dir / "graph.json"
            graph_path.write_text(
                json.dumps(
                    {
                        "directed": False,
                        "multigraph": False,
                        "graph": {},
                        "nodes": [
                            {
                                "id": "a",
                                "label": "cache_loader",
                                "docstring": "cache vectors",
                            }
                        ],
                        "links": [],
                        "hyperedges": [],
                    }
                ),
                encoding="utf-8",
            )
            service = ModelService(
                ServiceManager(),
                idle_timeout_seconds=60,
                clock=lambda: now[0],
            )
            service.search({"graph": str(graph_path), "query": "cache", "neighbors": 0})
            self.assertEqual(service.status()["node_count"], 1)

            now[0] = 159.999
            self.assertEqual(service.tick(), [])
            self.assertIsNotNone(service.status())

            now[0] = 160.0
            self.assertEqual(service.tick(), ["index"])
            self.assertIsNone(service.status())

    def test_status_synchronously_hides_index_at_deadline(self):
        now = [10.0]
        with tempfile.TemporaryDirectory() as temporary:
            graph_dir = Path(temporary) / "graphify-out"
            graph_dir.mkdir()
            graph_path = graph_dir / "graph.json"
            graph_path.write_text(
                json.dumps(
                    {
                        "directed": False,
                        "multigraph": False,
                        "graph": {},
                        "nodes": [{"id": "a", "label": "cache", "docstring": "cache"}],
                        "links": [],
                        "hyperedges": [],
                    }
                ),
                encoding="utf-8",
            )
            service = ModelService(
                ServiceManager(), idle_timeout_seconds=60, clock=lambda: now[0]
            )
            service.search({"graph": str(graph_path), "query": "cache", "neighbors": 0})
            now[0] = 70.0
            self.assertIsNone(service.status())

    def test_status_synchronously_invalidates_changed_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            graph_dir = Path(temporary) / "graphify-out"
            graph_dir.mkdir()
            graph_path = graph_dir / "graph.json"
            graph_path.write_text(
                json.dumps(
                    {
                        "directed": False,
                        "multigraph": False,
                        "graph": {},
                        "nodes": [{"id": "a", "label": "cache", "docstring": "cache"}],
                        "links": [],
                        "hyperedges": [],
                    }
                ),
                encoding="utf-8",
            )
            service = ModelService(ServiceManager())
            service.search({"graph": str(graph_path), "query": "cache", "neighbors": 0})
            metadata_path = graph_dir / "cache" / "embeddings.json"
            metadata_path.write_text(
                metadata_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
            )
            self.assertIsNone(service.status())

    def test_search_does_not_reuse_index_at_deadline(self):
        from unittest.mock import patch

        from graphify_embeddings.index import EmbeddingIndex

        now = [10.0]
        with tempfile.TemporaryDirectory() as temporary:
            graph_dir = Path(temporary) / "graphify-out"
            graph_dir.mkdir()
            graph_path = graph_dir / "graph.json"
            graph_path.write_text(
                json.dumps(
                    {
                        "directed": False,
                        "multigraph": False,
                        "graph": {},
                        "nodes": [{"id": "a", "label": "cache", "docstring": "cache"}],
                        "links": [],
                        "hyperedges": [],
                    }
                ),
                encoding="utf-8",
            )
            service = ModelService(
                ServiceManager(), idle_timeout_seconds=60, clock=lambda: now[0]
            )
            service.search({"graph": str(graph_path), "query": "cache", "neighbors": 0})
            now[0] = 70.0
            real_load = EmbeddingIndex.load
            loads = []

            def tracked_load(index):
                loads.append(index.vectors_path)
                return real_load(index)

            with patch.object(EmbeddingIndex, "load", tracked_load):
                service.search(
                    {"graph": str(graph_path), "query": "cache", "neighbors": 0}
                )
            self.assertEqual(len(loads), 1)

    def test_index_build_immediately_invalidates_resident_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            graph_dir = Path(temporary) / "graphify-out"
            graph_dir.mkdir()
            graph_path = graph_dir / "graph.json"
            graph_path.write_text(
                json.dumps(
                    {
                        "directed": False,
                        "multigraph": False,
                        "graph": {},
                        "nodes": [{"id": "a", "label": "cache", "docstring": "cache"}],
                        "links": [],
                        "hyperedges": [],
                    }
                ),
                encoding="utf-8",
            )
            service = ModelService(ServiceManager())
            service.search({"graph": str(graph_path), "query": "cache", "neighbors": 0})
            self.assertIsNotNone(service.status())
            service.index({"graph": str(graph_path), "force": False})
            self.assertIsNone(service.status())

    def test_changed_graph_and_cache_invalidate_resident_index(self):
        from graphify_embeddings.graph import GraphifyGraph
        from graphify_embeddings.index import EmbeddingIndex

        with tempfile.TemporaryDirectory() as temporary:
            graph_dir = Path(temporary) / "graphify-out"
            graph_dir.mkdir()
            graph_path = graph_dir / "graph.json"
            graph = {
                "directed": False,
                "multigraph": False,
                "graph": {},
                "nodes": [
                    {
                        "id": "a",
                        "label": "cache_loader",
                        "docstring": "cache vectors",
                    }
                ],
                "links": [],
                "hyperedges": [],
            }
            graph_path.write_text(json.dumps(graph), encoding="utf-8")
            manager = ServiceManager()
            service = ModelService(manager)
            first = service.search(
                {"graph": str(graph_path), "query": "cache", "neighbors": 0}
            )
            self.assertEqual(first["results"][0]["id"], "a")

            graph["nodes"].append(
                {"id": "b", "label": "gpu_runner", "docstring": "gpu model"}
            )
            graph_path.write_text(json.dumps(graph), encoding="utf-8")
            EmbeddingIndex(GraphifyGraph(graph_path)).build(
                manager.embedding, force=False, show_progress=False
            )

            second = service.search(
                {"graph": str(graph_path), "query": "gpu", "neighbors": 0}
            )
            self.assertEqual(second["results"][0]["id"], "b")
            self.assertEqual(service.status()["node_count"], 2)

    def test_cache_replacement_during_search_does_not_relabel_old_index(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temporary:
            graph_dir = Path(temporary) / "graphify-out"
            graph_dir.mkdir()
            graph_path = graph_dir / "graph.json"
            graph_path.write_text(
                json.dumps(
                    {
                        "directed": False,
                        "multigraph": False,
                        "graph": {},
                        "nodes": [{"id": "a", "label": "cache", "docstring": "cache"}],
                        "links": [],
                        "hyperedges": [],
                    }
                ),
                encoding="utf-8",
            )
            service = ModelService(ServiceManager())
            service.search({"graph": str(graph_path), "query": "cache", "neighbors": 0})
            original = service._resident_signature
            self.assertIsNotNone(original)
            replaced = tuple(
                (*signature[:1], signature[1] + 1, *signature[2:])
                for signature in original
            )
            with patch.object(
                service, "_index_signature", side_effect=[original, replaced]
            ):
                result = service.search(
                    {"graph": str(graph_path), "query": "cache", "neighbors": 0}
                )
            self.assertEqual(result["results"][0]["id"], "a")
            self.assertIsNone(service.status())

    def test_index_without_stable_disk_signature_is_not_cached(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temporary:
            graph_dir = Path(temporary) / "graphify-out"
            graph_dir.mkdir()
            graph_path = graph_dir / "graph.json"
            graph_path.write_text(
                json.dumps(
                    {
                        "directed": False,
                        "multigraph": False,
                        "graph": {},
                        "nodes": [{"id": "a", "label": "cache", "docstring": "cache"}],
                        "links": [],
                        "hyperedges": [],
                    }
                ),
                encoding="utf-8",
            )
            service = ModelService(ServiceManager())
            with patch.object(ModelService, "_index_signature", return_value=None):
                result = service.search(
                    {"graph": str(graph_path), "query": "cache", "neighbors": 0}
                )
            self.assertEqual(result["results"][0]["id"], "a")
            self.assertIsNone(service.status())

    def test_index_and_reranked_search_reuse_resident_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            graph_dir = root / "graphify-out"
            graph_dir.mkdir()
            graph = {
                "directed": False,
                "multigraph": False,
                "graph": {},
                "nodes": [
                    {"id": "a", "label": "cache_loader", "docstring": "cache vectors"},
                    {"id": "b", "label": "gpu_runner", "docstring": "gpu model"},
                ],
                "links": [],
                "hyperedges": [],
            }
            graph_path = graph_dir / "graph.json"
            graph_path.write_text(json.dumps(graph), encoding="utf-8")
            manager = ServiceManager()
            service = ModelService(manager)
            stats = service.index({"graph": str(graph_path), "force": False})
            self.assertEqual(stats["embedded"], 2)
            payload = service.search(
                {
                    "graph": str(graph_path),
                    "query": "cache",
                    "top_k": 2,
                    "candidate_k": 2,
                    "neighbors": 0,
                    "rerank": True,
                    "reranker_batch_size": 1,
                }
            )
            self.assertEqual(payload["results"][0]["id"], "b")
            self.assertEqual(manager.reranker_overrides, [True])
            self.assertFalse(manager.embedding.closed)
            self.assertFalse(manager.reranker.closed)


if __name__ == "__main__":
    unittest.main()
