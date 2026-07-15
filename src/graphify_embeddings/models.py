from __future__ import annotations

import gc
import hashlib
import importlib.util
import os
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np


DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-VL-Embedding-8B"
DEFAULT_EMBEDDING_REVISION = "2c4565515e0f265c6511776e7193b22c0968ddc7"
DEFAULT_EMBEDDING_WRAPPER_SHA256 = (
    "8ffa74a1a6bb759610c57865ea416fd4daf9936cb787520e1112a3e1d547f36a"
)
DEFAULT_RERANKER_MODEL = "Qwen/Qwen3-VL-Reranker-8B"
DEFAULT_RERANKER_REVISION = "b212dc8c91a8164aef1ea2de9c1a867611e75c04"
DEFAULT_RERANKER_WRAPPER_SHA256 = (
    "bd5d2f5d97fc4a738864d93f6b15d8850243e60da4484f3ea78867a46efdebd6"
)
DEFAULT_INSTRUCTION = "Retrieve code graph nodes relevant to the user's query."


def _imports():
    try:
        import torch
        from sentence_transformers import CrossEncoder, SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "GPU dependencies are missing. Install with: uv pip install -e '.[gpu]'"
        ) from exc
    return torch, SentenceTransformer, CrossEncoder


def resolve_device(requested: str = "auto") -> str:
    torch, _, _ = _imports()
    requested = str(requested or "auto").strip().lower()
    if requested == "auto":
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            torch.cuda.set_device(0)
            return "cuda:0"
        return "cpu"
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA requested ({requested}) but CUDA is unavailable")
        index = int(requested.split(":", 1)[1]) if ":" in requested else 0
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device {index} does not exist; found {torch.cuda.device_count()} device(s)"
            )
        torch.cuda.set_device(index)
        return f"cuda:{index}"
    if requested == "cpu":
        return "cpu"
    raise ValueError(f"Unsupported device: {requested}")


def resolve_dtype(dtype: str, device: str):
    torch, _, _ = _imports()
    name = str(dtype or "bf16").lower()
    if device == "cpu":
        return torch.float32
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[name]


class QwenEmbedder:
    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        device: str = "auto",
        dtype: str = "bf16",
        batch_size: int = 4,
        instruction: str = DEFAULT_INSTRUCTION,
        local_files_only: bool = False,
    ):
        torch, SentenceTransformer, _ = _imports()
        self.torch = torch
        self.model_name = model_name
        self.device = resolve_device(device)
        self.requested_dtype = str(dtype).lower()
        self.dtype = resolve_dtype(dtype, self.device)
        self.batch_size = max(1, int(batch_size))
        self.instruction = instruction
        self.backend = "sentence_transformers"
        self.revision = (
            DEFAULT_EMBEDDING_REVISION
            if model_name == DEFAULT_EMBEDDING_MODEL
            else None
        )
        self.wrapper_sha256 = None

        # Exact backend used by the Comfy custom node: the official model-snapshot
        # Qwen3VLEmbedder and its process([{"text": ...}]) API. No llama.cpp,
        # Ollama, or HTTP server is involved.
        if (
            model_name == DEFAULT_EMBEDDING_MODEL
            or Path(model_name).expanduser().is_dir()
        ):
            model_path = self._resolve_model_path(model_name, local_files_only)
            script = model_path / "scripts" / "qwen3_vl_embedding.py"
            if script.is_file():
                if self.device == "cpu":
                    raise RuntimeError(
                        "The official Qwen3-VL embedding wrapper used by the Comfy node "
                        "requires CUDA; choose --device cuda:N"
                    )
                self.wrapper_sha256 = hashlib.sha256(script.read_bytes()).hexdigest()
                if (
                    model_name == DEFAULT_EMBEDDING_MODEL
                    and self.wrapper_sha256 != DEFAULT_EMBEDDING_WRAPPER_SHA256
                ):
                    raise RuntimeError(
                        "Pinned Qwen embedding wrapper failed SHA-256 verification"
                    )
                wrapper_class = self._load_wrapper(script)
                try:
                    self.model = wrapper_class(
                        model_name_or_path=str(model_path),
                        torch_dtype=self.dtype,
                        local_files_only=True,
                    )
                except Exception:
                    gc.collect()
                    if self.torch.cuda.is_available():
                        self.torch.cuda.empty_cache()
                    raise
                self.backend = "official_vl_wrapper"
                return

        # Explicit alternative models such as Qwen3-Embedding-8B retain the
        # standard SentenceTransformers path.
        self.model = SentenceTransformer(
            model_name,
            device=self.device,
            model_kwargs={"torch_dtype": self.dtype},
            local_files_only=local_files_only,
        )

    @staticmethod
    def _resolve_model_path(model_name: str, local_files_only: bool) -> Path:
        local = Path(model_name).expanduser()
        if local.is_dir():
            return local.resolve()
        if model_name != DEFAULT_EMBEDDING_MODEL:
            raise ValueError(
                "Official VL wrapper loading is restricted to "
                f"{DEFAULT_EMBEDDING_MODEL!r} or an explicit local model directory"
            )
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "huggingface-hub is required for Qwen VL embedding"
            ) from exc
        return Path(
            snapshot_download(
                repo_id=model_name,
                revision=DEFAULT_EMBEDDING_REVISION,
                local_files_only=local_files_only,
            )
        ).resolve()

    @staticmethod
    def _load_wrapper(script: Path):
        module_name = "graphify_embeddings._qwen3_vl_embedding"
        spec = importlib.util.spec_from_file_location(module_name, script)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load official Qwen embedding wrapper: {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        wrapper = getattr(module, "Qwen3VLEmbedder", None)
        if wrapper is None:
            raise RuntimeError(f"Qwen3VLEmbedder class missing in {script}")
        return wrapper

    def cache_identity(self) -> dict[str, str | None]:
        return {
            "model": self.model_name,
            "backend": self.backend,
            "instruction": self.instruction,
            "revision": self.revision,
            "wrapper_sha256": self.wrapper_sha256,
            "dtype": self.requested_dtype,
        }

    def encode(
        self, texts: Sequence[str], *, show_progress: bool = False
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        if self.backend == "official_vl_wrapper":
            batches = []
            text_list = list(texts)
            for start in range(0, len(text_list), self.batch_size):
                chunk = text_list[start : start + self.batch_size]
                embeddings = self.model.process(
                    [{"text": text, "instruction": self.instruction} for text in chunk],
                    normalize=True,
                )
                if hasattr(embeddings, "detach"):
                    embeddings = embeddings.detach().float().cpu().numpy()
                batches.append(np.asarray(embeddings, dtype=np.float32))
            vectors = np.concatenate(batches, axis=0)
        else:
            vectors = self.model.encode(
                list(texts),
                prompt=self.instruction,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)

    def close(self) -> None:
        model = getattr(self, "model", None)
        self.model = None
        if model is not None:
            del model
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class QwenReranker:
    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        *,
        device: str = "auto",
        dtype: str = "bf16",
        instruction: str = DEFAULT_INSTRUCTION,
        local_files_only: bool = False,
        max_length: int = 4096,
    ):
        torch, _, CrossEncoder = _imports()
        self.torch = torch
        self.model_name = model_name
        self.device = resolve_device(device)
        self.requested_dtype = str(dtype).lower()
        self.dtype = resolve_dtype(dtype, self.device)
        self.instruction = instruction
        self.backend = "cross_encoder"
        self.revision = (
            DEFAULT_RERANKER_REVISION if model_name == DEFAULT_RERANKER_MODEL else None
        )

        # sentence-transformers 5.5 cannot reconstruct Qwen's saved LogitScore
        # module from the 5.4 model artifact (missing true_token_id). The official
        # model snapshot ships a stable process() wrapper, which is also the path
        # used by the working ComfyUI implementation.
        if (
            model_name == DEFAULT_RERANKER_MODEL
            or Path(model_name).expanduser().is_dir()
        ):
            model_path = self._resolve_model_path(model_name, local_files_only)
            script = model_path / "scripts" / "qwen3_vl_reranker.py"
            if script.is_file():
                if self.device == "cpu":
                    raise RuntimeError(
                        "The official Qwen3-VL reranker wrapper used by the Comfy node "
                        "requires CUDA; choose --device cuda:N"
                    )
                wrapper_sha256 = hashlib.sha256(script.read_bytes()).hexdigest()
                if (
                    model_name == DEFAULT_RERANKER_MODEL
                    and wrapper_sha256 != DEFAULT_RERANKER_WRAPPER_SHA256
                ):
                    raise RuntimeError(
                        "Pinned Qwen reranker wrapper failed SHA-256 verification"
                    )
                wrapper_class = self._load_wrapper(script)
                load_path = self._processor_compatible_view(model_path)
                try:
                    self.model = wrapper_class(
                        model_name_or_path=str(load_path),
                        max_length=max_length,
                        torch_dtype=self.dtype,
                        local_files_only=True,
                    )
                except Exception:
                    model_view = getattr(self, "_model_view", None)
                    self._model_view = None
                    if model_view is not None:
                        model_view.cleanup()
                    gc.collect()
                    if self.torch.cuda.is_available():
                        self.torch.cuda.empty_cache()
                    raise
                self.backend = "official_vl_wrapper"
                return

        self.model = CrossEncoder(
            model_name,
            device=self.device,
            model_kwargs={"torch_dtype": self.dtype},
            local_files_only=local_files_only,
            max_length=max_length,
        )

    @staticmethod
    def _resolve_model_path(model_name: str, local_files_only: bool) -> Path:
        local = Path(model_name).expanduser()
        if local.is_dir():
            return local.resolve()
        if model_name != DEFAULT_RERANKER_MODEL:
            raise ValueError(
                "Official VL wrapper loading is restricted to "
                f"{DEFAULT_RERANKER_MODEL!r} or an explicit local model directory"
            )
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "huggingface-hub is required for Qwen VL reranking"
            ) from exc
        return Path(
            snapshot_download(
                repo_id=model_name,
                revision=DEFAULT_RERANKER_REVISION,
                local_files_only=local_files_only,
            )
        ).resolve()

    def _processor_compatible_view(self, model_path: Path) -> Path:
        """Hide legacy chat_template.json when modern template files coexist.

        Transformers 5 rejects snapshots containing both formats. Qwen's current
        reranker snapshot contains both, so create a temporary symlink view instead
        of mutating the shared Hugging Face cache.
        """
        legacy = model_path / "chat_template.json"
        additional = model_path / "additional_chat_templates"
        modern = model_path / "chat_template.jinja"
        if not (legacy.is_file() and additional.is_dir() and modern.is_file()):
            self._model_view = None
            return model_path
        self._model_view = tempfile.TemporaryDirectory(prefix="graphify-qwen-reranker-")
        view = Path(self._model_view.name)
        for entry in model_path.iterdir():
            if entry.name == "chat_template.json":
                continue
            os.symlink(entry, view / entry.name, target_is_directory=entry.is_dir())
        return view

    @staticmethod
    def _load_wrapper(script: Path):
        module_name = "graphify_embeddings._qwen3_vl_reranker"
        spec = importlib.util.spec_from_file_location(module_name, script)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load official Qwen reranker wrapper: {script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        wrapper = getattr(module, "Qwen3VLReranker", None)
        if wrapper is None:
            raise RuntimeError(f"Qwen3VLReranker class missing in {script}")
        return wrapper

    def score(
        self, query: str, documents: Sequence[str], batch_size: int = 1
    ) -> np.ndarray:
        if not documents:
            return np.empty((0,), dtype=np.float32)
        if self.backend == "official_vl_wrapper":
            scores = self.model.process(
                {
                    "instruction": self.instruction,
                    "query": {"text": query},
                    "documents": [{"text": document} for document in documents],
                }
            )
            return np.asarray(scores, dtype=np.float32).reshape(-1)

        pairs = [(query, document) for document in documents]
        try:
            scores = self.model.predict(
                pairs,
                prompt=self.instruction,
                activation_fn=self.torch.nn.Sigmoid(),
                batch_size=max(1, int(batch_size)),
                show_progress_bar=False,
            )
        except TypeError:
            scores = self.model.predict(
                pairs,
                activation_fn=self.torch.nn.Sigmoid(),
                batch_size=max(1, int(batch_size)),
                show_progress_bar=False,
            )
        return np.asarray(scores, dtype=np.float32).reshape(-1)

    def close(self) -> None:
        model = getattr(self, "model", None)
        self.model = None
        if model is not None:
            del model
        gc.collect()
        model_view = getattr(self, "_model_view", None)
        self._model_view = None
        if model_view is not None:
            model_view.cleanup()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
