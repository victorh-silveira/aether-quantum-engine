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
"$PY" -u app/scripts/operations/run_launch_train_tf_pipeline.py "$@"
if ! "$PY" -u app/scripts/operations/check_dl_deploy_gate.py --allow-unqualified; then
  echo "[AVISO] TCN sem qualificacao de deploy; checkpoint local permanece sujeito ao teto de risco."
fi
if ! "$PY" -u app/scripts/operations/ensure_timescale.py; then
  echo "[AVISO] Timescale seed indisponivel; meta tentara API Deriv."
fi
"$PY" -u app/scripts/operations/train_meta_classifier.py --trials 60 --bars 5000 --source auto --candidate-on-low-quality --export-min-zscore -0.05 --export-min-ir -0.50
if "$PY" -u app/scripts/operations/check_dl_deploy_gate.py --with-meta --allow-unqualified; then
  echo "[AETHER] TCN + meta Rise/Fall qualificados (operacao sob teto de stake)."
else
  if [[ -f "$REPO_ROOT/infra/docker/meta-models/meta_lgbm.pkl" ]]; then
    echo "[AVISO] Meta exportado em meta-models, mas gate conjunto TCN + meta nao qualificado."
  else
    echo "[AVISO] Meta nao exportado; somente candidato diagnostico. Gate conjunto TCN + meta nao qualificado."
  fi
fi
