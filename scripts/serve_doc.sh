#!/bin/sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
SITE_DIR="${ROOT_DIR}/build/docs/site"
PORT="${1:-8000}"
VENV_DIR="${ROOT_DIR}/env"

if [ ! -d "${SITE_DIR}" ]; then
  echo "Documentation site not found in ${SITE_DIR}." >&2
  echo "Run ./scripts/build_doc.sh first." >&2
  exit 1
fi

# activate venv if present
if [ -d "${VENV_DIR}" ]; then
  # shellcheck source=/dev/null
  . "${VENV_DIR}/bin/activate"
fi

export SITE_DIR

# Prefer 'uv' static server if available (try common invocation patterns),
# otherwise fall back to uvicorn ASGI server serving the Starlette app.
if command -v uv >/dev/null 2>&1; then
  echo "Found 'uv' binary; serving ${SITE_DIR} with uv on port ${PORT}"
  # Inspect help output to pick a compatible invocation and exec it.
  help_output=$(uv --help 2>&1 || true)
  fallback() {
    echo "Falling back to Python HTTP server on port ${PORT}"
    if command -v python3 >/dev/null 2>&1; then
      exec python3 -m http.server "${PORT}" --directory "${SITE_DIR}"
    else
      exec python -m http.server "${PORT}" --directory "${SITE_DIR}"
    fi
  }

  if echo "$help_output" | grep -q -- '--root'; then
    uv --port "${PORT}" --root "${SITE_DIR}" || fallback
  elif echo "$help_output" | grep -q -- 'serve'; then
    uv serve "${SITE_DIR}" --port "${PORT}" || fallback
  elif echo "$help_output" | grep -q -- '--port'; then
    uv --port "${PORT}" "${SITE_DIR}" || fallback
  else
    # Last resort: try running with site dir only, then fallback
    uv "${SITE_DIR}" || fallback
  fi
else
  echo "'uv' not found in PATH. Install it with:"
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi
