#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -x "$project_dir/.venv/bin/python" ]]; then
    if command -v conda >/dev/null 2>&1; then
        CONDA_PKGS_DIRS="$project_dir/.cache/conda-pkgs" conda env create --yes --prefix "$project_dir/.venv" --file "$project_dir/environment.yml"
    else
        python3 -m venv "$project_dir/.venv"
    fi
fi
PIP_CACHE_DIR="$project_dir/.cache/pip" "$project_dir/.venv/bin/python" -m pip install --requirement "$project_dir/requirements.lock.txt"
PIP_CACHE_DIR="$project_dir/.cache/pip" "$project_dir/.venv/bin/python" -m pip install --editable "$project_dir[test,visualization]"
