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
NEW="$BIN_DIR/siftsc"
echo "✅ SiftSC installed: $("$NEW" --version)"

# An older copy installed with pip (for example into a conda or system Python) can sit
# earlier on PATH and shadow the new command. Remove such copies of this same tool and
# say so; anything that cannot be removed is reported with the exact fix.
old_ifs=$IFS
IFS=:
for dir in $PATH; do
  candidate="$dir/siftsc"
  [ -x "$candidate" ] || continue
  [ "$candidate" = "$NEW" ] && break
  interpreter="$(sed -n '1s/^#!//p' "$candidate" | tr -d '\r')"
  case "$interpreter" in
    *python*)
      if "$interpreter" -m pip uninstall -y -q siftsc >/dev/null 2>&1 && [ ! -e "$candidate" ]; then
        echo "🧹 Removed an older siftsc from $candidate so the new command is used"
        echo "   (reinstall it any time with: $interpreter -m pip install siftsc)"
      else
        echo "⚠️  An older siftsc at $candidate comes first on your PATH."
        echo "   Remove it with:  $interpreter -m pip uninstall siftsc"
      fi
      ;;
    *)
      echo "⚠️  Another siftsc at $candidate comes first on your PATH; remove it or put $BIN_DIR first."
      ;;
  esac
done
IFS=$old_ifs
hash -r 2>/dev/null || true

if [ "$(command -v siftsc 2>/dev/null)" = "$NEW" ]; then
  echo "🚀 Ready. Type:  siftsc"
else
  echo "🚀 Open a new terminal and type:  siftsc"
  echo "   (if the command is not found, run:  export PATH=\"$BIN_DIR:\$PATH\")"
fi
