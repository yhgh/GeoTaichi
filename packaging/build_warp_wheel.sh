#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if ! command -v python >/dev/null 2>&1; then
  echo "[error] Python is required to build GeoWarp." >&2
  exit 1
fi

if [[ ! -f pyproject.toml ]]; then
  echo "[error] pyproject.toml not found. Run this script from the repository root." >&2
  exit 1
fi

python -m pip install --upgrade pip >/dev/null
python -m pip install --upgrade build twine >/dev/null

rm -rf build dist
python -m build

twine check dist/*

echo "Artifacts generated under dist/:"
ls -1 dist
