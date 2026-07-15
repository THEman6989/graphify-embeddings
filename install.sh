#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV="${GRAPHIFY_EMBEDDINGS_VENV:-$ROOT/.venv}"
PYTHON_VERSION="${GRAPHIFY_EMBEDDINGS_PYTHON:-3.11}"
FLASH_MODE="auto"

usage() {
  printf '%s\n' "Usage: ./install.sh [--no-flash-attn|--require-flash-attn]"
}

while (($#)); do
  case "$1" in
    --no-flash-attn) FLASH_MODE="off" ;;
    --require-flash-attn) FLASH_MODE="required" ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if ! command -v uv >/dev/null 2>&1; then
  printf '%s\n' "error: uv is required: https://docs.astral.sh/uv/" >&2
  exit 1
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv --python "$PYTHON_VERSION" "$VENV"
else
  "$VENV/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 11)'
  printf 'Reusing virtual environment: %s\n' "$VENV"
fi
VIRTUAL_ENV="$VENV" uv sync --project "$ROOT" --locked --extra gpu --no-editable --active

FLASH_INSTALLED=false
if [[ "$FLASH_MODE" == "off" ]]; then
  uv pip uninstall --python "$VENV/bin/python" flash-attn >/dev/null 2>&1 || true
elif [[ "$FLASH_MODE" == "auto" ]]; then
  if uv pip install --python "$VENV/bin/python" --only-binary=:all: flash-attn; then
    FLASH_INSTALLED=true
  else
    printf '%s\n' "warning: no compatible flash-attn wheel; using PyTorch SDPA" >&2
  fi
elif [[ "$FLASH_MODE" == "required" ]]; then
  uv pip install --python "$VENV/bin/python" packaging ninja wheel setuptools
  if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" || -z "${FLASH_ATTN_CUDA_ARCHS:-}" ]]; then
    ARCHITECTURES="$($VENV/bin/python - <<'PY'
import torch
capabilities = sorted({torch.cuda.get_device_capability(index) for index in range(torch.cuda.device_count())})
torch_archs = ";".join(f"{major}.{minor}" for major, minor in capabilities)
flash_archs = set()
for major, _minor in capabilities:
    if major >= 12:
        flash_archs.add("120")
    elif major >= 10:
        flash_archs.add("100")
    elif major == 9:
        flash_archs.add("90")
    elif major == 8:
        flash_archs.add("80")
print(f"{torch_archs}|{';'.join(sorted(flash_archs, key=int))}")
PY
)"
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-${ARCHITECTURES%%|*}}"
    FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-${ARCHITECTURES#*|}}"
    export TORCH_CUDA_ARCH_LIST FLASH_ATTN_CUDA_ARCHS
  fi
  printf 'Torch CUDA architectures: %s\n' "$TORCH_CUDA_ARCH_LIST"
  printf 'FlashAttention CUDA architectures: %s\n' "$FLASH_ATTN_CUDA_ARCHS"
  if MAX_JOBS="${MAX_JOBS:-4}" uv pip install --python "$VENV/bin/python" \
      --no-build-isolation flash-attn; then
    FLASH_INSTALLED=true
  else
    printf '%s\n' "error: flash-attn source build failed" >&2
    exit 1
  fi
fi

"$VENV/bin/python" - <<'PY'
import importlib.util
import json
import sys
import torch
import transformers

if sys.prefix == sys.base_prefix:
    raise SystemExit("not running inside the dedicated virtual environment")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the dedicated virtual environment")
print(json.dumps({
    "python": sys.executable,
    "prefix": sys.prefix,
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "cuda_devices": torch.cuda.device_count(),
    "flash_attn": importlib.util.find_spec("flash_attn") is not None,
}, indent=2))
PY

BIN_DIR="${HOME}/.local/bin"
TARGET="$BIN_DIR/graphify-embeddings"
TEMP_TARGET="$TARGET.tmp.$$"
mkdir -p "$BIN_DIR"
ln -s "$VENV/bin/graphify-embeddings" "$TEMP_TARGET"
mv -Tf "$TEMP_TARGET" "$TARGET"

CONFIG="${XDG_CONFIG_HOME:-$HOME/.config}/graphify-embeddings/config.toml"
if [[ ! -f "$CONFIG" ]]; then
  "$TARGET" config init >/dev/null
fi

printf 'Installed: %s -> %s\n' "$TARGET" "$VENV/bin/graphify-embeddings"
printf 'Config: %s\n' "$CONFIG"
printf 'FlashAttention package installed: %s\n' "$FLASH_INSTALLED"
