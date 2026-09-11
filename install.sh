#!/bin/sh
# SiftSC one-line installer for Apple-Silicon Macs.
#
#   curl -fsSL https://raw.githubusercontent.com/heywanrong/SiftSC/main/install.sh | sh
#
# It installs uv (a Python tool manager that also fetches Python itself) when it is
# missing, then installs SiftSC with the MLX backend and the full-screen interface as
# an isolated tool, and puts the `siftsc` command on your PATH. Nothing else on the
# machine is touched. Environment overrides: SIFTSC_SOURCE (package spec),
# SIFTSC_PYTHON (default 3.12), SIFTSC_NO_SHELL_UPDATE=1 (do not edit shell files).
set -eu

SOURCE="${SIFTSC_SOURCE:-siftsc[mlx] @ git+https://github.com/heywanrong/SiftSC.git}"
PYTHON="${SIFTSC_PYTHON:-3.12}"

if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
  echo "❌ SiftSC's bundled model runner needs an Apple-Silicon Mac (M1 or newer)." >&2
  echo "   The NumPy routing core works elsewhere through the Python API with a custom backend." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "📦 Installing uv (Python tool manager) …"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "📦 Installing SiftSC (Python $PYTHON is fetched automatically if needed) …"
uv tool install --force --quiet --python "$PYTHON" "$SOURCE"

if [ -z "${SIFTSC_NO_SHELL_UPDATE:-}" ]; then
  uv tool update-shell >/dev/null 2>&1 || true
fi

BIN_DIR="$(uv tool dir --bin 2>/dev/null || echo "$HOME/.local/bin")"
echo "✅ SiftSC installed: $("$BIN_DIR/siftsc" --version)"
echo "🚀 Open a new terminal and type:  siftsc"
echo "   (if the command is not found, run:  export PATH=\"$BIN_DIR:\$PATH\")"
