#!/usr/bin/env bash
# Install OCBench from this repository with every optional extra:
# train (JAX, Flax, wandb, …), mjwarp, data (LeRobot), and dev (ruff).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
VENV="$ROOT/.venv"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but was not found in PATH." >&2
  echo "See https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

if [[ ! -d "$VENV" ]]; then
  echo "Creating virtual environment at .venv (Python 3.10+)…"
  uv venv --python '>=3.10' "$VENV"
else
  echo "Using existing virtual environment at .venv (will not recreate or replace it)…"
fi

echo "Installing/updating ocbench and optional dependencies…"
uv pip install --python "$VENV/bin/python" -e ".[all]"

if [[ "$(uname -s)" == "Linux" ]]; then
  echo "Configuring the FFmpeg libraries bundled with PyAV for TorchCodec…"
  AV_LIBS="$("$VENV/bin/python" - <<'PY'
import pathlib

import av

av_libs = pathlib.Path(av.__file__).resolve().parent.parent / 'av.libs'
if not av_libs.is_dir():
    raise SystemExit(f'PyAV shared-library directory was not found: {av_libs}')
print(av_libs)
PY
)"
  TORCHCODEC_FFMPEG_LIB="$VENV/lib/torchcodec-ffmpeg"
  mkdir -p "$TORCHCODEC_FFMPEG_LIB"

  # PyAV wheels bundle FFmpeg with hashed SONAMEs. Make all bundled libraries
  # visible, then add the standard ABI names that TorchCodec loads with dlopen.
  find "$TORCHCODEC_FFMPEG_LIB" -mindepth 1 -maxdepth 1 -type l -delete
  shopt -s nullglob
  for library in "$AV_LIBS"/*.so*; do
    ln -s "$library" "$TORCHCODEC_FFMPEG_LIB/$(basename "$library")"
  done
  for component in avutil avcodec avformat avdevice avfilter swscale swresample; do
    libraries=("$AV_LIBS/lib${component}-"*.so.*)
    if (( ${#libraries[@]} != 1 )); then
      echo "Expected one bundled lib${component}, found ${#libraries[@]} in $AV_LIBS" >&2
      exit 1
    fi
    library="${libraries[0]}"
    version="${library##*.so.}"
    major="${version%%.*}"
    ln -s "$library" "$TORCHCODEC_FFMPEG_LIB/lib${component}.so.${major}"
  done
  shopt -u nullglob

  ACTIVATE_HOOK="$VENV/bin/activate-ocbench"
  cat >"$ACTIVATE_HOOK" <<'EOF'
# Give TorchCodec access to the shared FFmpeg libraries bundled with PyAV.
_ocbench_ffmpeg_lib="$VIRTUAL_ENV/lib/torchcodec-ffmpeg"
case ":${LD_LIBRARY_PATH-}:" in
  *":${_ocbench_ffmpeg_lib}:"*) ;;
  *) export LD_LIBRARY_PATH="${_ocbench_ffmpeg_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
esac
unset _ocbench_ffmpeg_lib
EOF

  ACTIVATE_MARKER='# Load OCBench environment additions.'
  if ! grep -Fq "$ACTIVATE_MARKER" "$VENV/bin/activate"; then
    cat >>"$VENV/bin/activate" <<'EOF'

# Load OCBench environment additions.
if [ -f "$VIRTUAL_ENV/bin/activate-ocbench" ]; then
    . "$VIRTUAL_ENV/bin/activate-ocbench"
fi
EOF
  fi

  echo "Checking TorchCodec with the local FFmpeg libraries…"
  LD_LIBRARY_PATH="$TORCHCODEC_FFMPEG_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$VENV/bin/python" - <<'PY'
import torch
import torchcodec
from torchcodec.decoders import VideoDecoder
from torchcodec.encoders import VideoEncoder

print(f'TorchCodec {torchcodec.__version__} loaded with PyTorch {torch.__version__}.')
PY
fi

echo "Done. Activate with: source .venv/bin/activate"
echo "Extras included: train, mjwarp, data (lerobot[dataset]), dev."
