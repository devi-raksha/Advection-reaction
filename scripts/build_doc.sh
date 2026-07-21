#!/bin/bash

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
# Prefer a repository-provided venv (e.g. `.venv`) if present (CI may create this).
if [ -d "${ROOT_DIR}/.venv" ]; then
  VENV_DIR="${ROOT_DIR}/.venv"
else
  VENV_DIR="${ROOT_DIR}/env"
fi
REQUIREMENTS_FILE="${ROOT_DIR}/doc/requirements.txt"
BUILD_DIR="${ROOT_DIR}/build/docs"
DOXYGEN_OUT="${BUILD_DIR}/doxygen"
SITE_DIR="${BUILD_DIR}/site"
DOXYFILE="${ROOT_DIR}/doc/Doxyfile"
DOCS_SOURCE_DIR="${ROOT_DIR}/doc"
API_STUB_DIR="${DOCS_SOURCE_DIR}/api"
MKDOCS_SITE_DIR="${ROOT_DIR}/site"

if [ -d "$HOME/anaconda3/bin" ]; then
  echo "Anaconda3 found."
  PATH="$HOME/anaconda3/bin:$PATH"
  export PATH
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found in PATH." >&2
  exit 1
fi


# doxygen is optional: only run if the environment variable RUN_DOXYGEN=1.
# This avoids heavy Doxygen/Sphinx processing unless explicitly requested.
if [ -f "${DOXYFILE}" ] && [ "${RUN_DOXYGEN:-0}" = "1" ]; then
  if ! command -v doxygen >/dev/null 2>&1; then
    echo "doxygen not found in PATH but Doxyfile exists and RUN_DOXYGEN=1. Skipping doxygen step." >&2
  fi
fi

if [ ! -d "${VENV_DIR}" ]; then
  # Use the stdlib venv module to create the virtual environment.
  python3 -m venv "${VENV_DIR}"
fi

. "${VENV_DIR}/bin/activate"

if [ ! -f "${REQUIREMENTS_FILE}" ]; then
  echo "${REQUIREMENTS_FILE} not found." >&2
  exit 1
fi

python3 -m pip install -r "${REQUIREMENTS_FILE}"

if ! command -v sphinx-build >/dev/null 2>&1; then
  echo "sphinx-build not found after installing documentation requirements." >&2
  exit 1
fi

mkdir -p "${BUILD_DIR}" "${DOXYGEN_OUT}" "${API_STUB_DIR}"
rm -rf "${SITE_DIR}" "${API_STUB_DIR}"
mkdir -p "${API_STUB_DIR}"

tmp_doxyfile="$(mktemp "${TMPDIR:-/tmp}/immersx-doxygen.XXXXXX")"
trap 'rm -f "${tmp_doxyfile}"' EXIT

sed \
  -e "s|^OUTPUT_DIRECTORY[[:space:]]*=.*$|OUTPUT_DIRECTORY = ${DOXYGEN_OUT}|" \
  -e "s|^GENERATE_HTML[[:space:]]*=.*$|GENERATE_HTML = NO|" \
  -e "s|^GENERATE_XML[[:space:]]*=.*$|GENERATE_XML = YES|" \
  -e "s|^HTML_OUTPUT[[:space:]]*=.*$|HTML_OUTPUT = html|" \
  -e "s|^XML_OUTPUT[[:space:]]*=.*$|XML_OUTPUT = xml|" \
  "${DOXYFILE}" > "${tmp_doxyfile}"

(
  if [ -f "${DOXYFILE}" ] && [ "${RUN_DOXYGEN:-0}" = "1" ]; then
    if command -v doxygen >/dev/null 2>&1; then
      (cd "${ROOT_DIR}" && doxygen "${tmp_doxyfile}")
    else
      echo "doxygen not found in PATH but RUN_DOXYGEN=1. Please install doxygen to generate API docs." >&2
      exit 127
    fi
  else
    echo "Skipping doxygen generation (set RUN_DOXYGEN=1 to enable)"
  fi
)

sphinx_cmd=(sphinx-build -b html "${DOCS_SOURCE_DIR}" "${SITE_DIR}")
# If STRICT_DOCS=1 then treat warnings as errors (for CI); otherwise allow warnings.
if [ "${STRICT_DOCS:-0}" = "1" ]; then
  sphinx_cmd=(sphinx-build -b html -W "${DOCS_SOURCE_DIR}" "${SITE_DIR}")
fi

"${sphinx_cmd[@]}"

echo "Documentation site generated in ${SITE_DIR}"
