#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1
PY="${AETHER_PYTHON:-/mnt/c/Users/victor-silveira/anaconda3/envs/deriv-api/python.exe}"
LOG_DIR="${AETHER_TRAIN_LOG_DIR:-/tmp/aether-train}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/launch-train.log"
exec > >(tee "$LOG") 2>&1

echo "[AETHER] launch-train Rise/Fall | TCN M5 | python=$PY"
"$PY" -u app/scripts/operations/sanitize_fresh_run.py
"$PY" -u app/train.py
"$PY" -u app/scripts/operations/check_dl_checkpoint.py
echo "[AETHER] Treino concluido; checkpoint TCN compativel sob teto de 1% da banca."
