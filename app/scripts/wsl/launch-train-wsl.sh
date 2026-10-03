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

echo "[AETHER] launch-train Rise/Fall | TCN M5 + meta LightGBM | python=$PY"
"$PY" -u app/scripts/operations/sanitize_fresh_run.py
(cd app && "$PY" -u -m scripts.operations.train_loss_classifier)
"$PY" -u app/train.py
"$PY" -u app/scripts/operations/check_dl_checkpoint.py
if ! "$PY" -u app/scripts/operations/ensure_timescale.py; then
  echo "[AVISO] Timescale seed indisponivel; meta tentara API Deriv."
fi
"$PY" -u app/scripts/operations/train_meta_classifier.py --trials 60 --bars 5000 --source auto --candidate-on-low-quality --export-min-zscore -0.05 --export-min-ir -0.50
echo "[AETHER] Treino concluido; checkpoint TCN compativel sob teto de 1% da banca."
