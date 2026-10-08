#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

printf 'Aether clean_env: caches pip/conda/torch (dev)\n'

if command -v conda >/dev/null 2>&1; then
  conda clean --all -y || true
fi

if command -v pip >/dev/null 2>&1; then
  pip cache purge || true
fi

rm -rf "${HOME}/.cache/pip"/* 2>/dev/null || true
rm -rf "${HOME}/.cache/torch/kernels"/* 2>/dev/null || true
rm -rf "${HOME}/.cache/torch/hub"/* 2>/dev/null || true

find "${REPO_ROOT}" -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
find "${REPO_ROOT}" -type d -name '.pytest_cache' -prune -exec rm -rf {} + 2>/dev/null || true
find "${REPO_ROOT}" -type d -name '.ruff_cache' -prune -exec rm -rf {} + 2>/dev/null || true

printf 'clean_env: concluido (nao remove volumes Docker nem data/)\n'
printf 'WSL trimestral: docker system prune; sudo fstrim /; compactar ext4.vhdx no Windows apos wsl --shutdown (diskpart manual; nao automatizado)\n'
