#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${MCX_OLD_VENV:-${HOME}/venvs/oldver}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install \
  -r "${SCRIPT_DIR}/requirements-old.txt"
"${VENV_DIR}/bin/python" -c \
  "import qiskit, pyzx; print('qiskit', qiskit.__version__, 'pyzx', pyzx.__version__)"
printf 'old compiler environment: %s\n' "${VENV_DIR}/bin/python"
