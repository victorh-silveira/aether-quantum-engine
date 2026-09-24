#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"
PY="${AETHER_PYTHON:-/mnt/c/Users/victor-silveira/anaconda3/envs/deriv-api/python.exe}"
LOG_DIR="${AETHER_TRAIN_LOG_DIR:-/tmp/aether-train}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/launch-train.log"
exec > >(tee "$LOG") 2>&1

echo "[AETHER] launch-train WSL | python=$PY | root=$REPO_ROOT"
echo "[AETHER] 0/5 sanitize"
"$PY" app/scripts/operations/sanitize_fresh_run.py
echo "[AETHER] 0b/5 loss-classifier bootstrap"
(cd app && "$PY" -m scripts.operations.train_loss_classifier)
echo "[AETHER] 1/5 treino TCN (horizon_sweep off -> app/train.py)"
"$PY" app/scripts/operations/run_launch_train_tf_pipeline.py
echo "[AETHER] 1b/5 gate deploy/settle"
DEPLOY_READY=1
if ! "$PY" app/scripts/operations/check_dl_deploy_gate.py; then
  DEPLOY_READY=0
  echo "[AVISO] Checkpoint TCN sem qualificacao OOS; runtime local sujeito a teto de stake."
fi
echo "[AETHER] 2/5 Timescale seed"
if ! "$PY" app/scripts/operations/ensure_timescale.py; then
  echo "[AVISO] Timescale seed falhou; meta usara API Deriv."
fi
echo "[AETHER] 3/5 meta LightGBM"
"$PY" app/scripts/operations/train_meta_classifier.py --trials 60 --bars 5000 --source auto --candidate-on-low-quality
if [ "$DEPLOY_READY" -eq 1 ] && [ -f infra/docker/meta-models/meta_lgbm.pkl ]; then
  echo "[SUCESSO] launch-train OK (TCN + meta; deploy aprovado)."
elif [ "$DEPLOY_READY" -eq 1 ]; then
  echo "[AVISO] TCN aprovado, mas meta apenas candidato; nao faca rebuild para trading."
else
  echo "[SUCESSO] Treino TCN + meta concluido; TCN nao qualificado, checkpoint local opera com teto de 0,1% em DEMO e REAL."
fi
echo "EXIT:0"
