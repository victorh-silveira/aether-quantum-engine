#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/docker-ui.sh"
source "${SCRIPT_DIR}/compose-lib.sh"

MODE="${1:-full}"
MODELS_DIR="${SCRIPT_DIR}/loss-models"
SERVICE="aether-loss-classifier"

cd "${REPO_ROOT}"

if [ ! -f .env ]; then
  cp .env.example .env
fi

mapfile -t COMPOSE_FLAGS < <(compose_args)
COMPOSE=(docker compose "${COMPOSE_FLAGS[@]}")

clear_models() {
  mkdir -p "${MODELS_DIR}"
  removed=0
  if [ -f "${MODELS_DIR}/learn_buffer.pkl" ]; then
    rm -f "${MODELS_DIR}/learn_buffer.pkl"
    removed=$((removed + 1))
  fi
  shopt -s nullglob
  for path in "${MODELS_DIR}"/*.pkl; do
    rm -f "${path}"
    removed=$((removed + 1))
  done
  shopt -u nullglob
  docker_ui_ok "removidos=${removed}"
}

stop_loss_if_running() {
  if "${COMPOSE[@]}" ps -q "${SERVICE}" 2>/dev/null | grep -q .; then
    "${COMPOSE[@]}" stop "${SERVICE}"
    docker_ui_ok "parado"
  else
    docker_ui_info "container ja parado"
  fi
}

validate_health_seeded() {
  payload="$(curl -sf "http://127.0.0.1:8006/health" 2>/dev/null || true)"
  if [ -z "${payload}" ]; then
    docker_ui_fail "Loss-classifier" "/health vazio"
    exit 1
  fi
  printf '%s\n' "${payload}"
  if ! printf '%s' "${payload}" | grep -q '"ready"[[:space:]]*:[[:space:]]*true'; then
    docker_ui_fail "Loss-classifier" "ready=false"
    exit 1
  fi
  if ! printf '%s' "${payload}" | grep -q '"model_loaded"[[:space:]]*:[[:space:]]*true'; then
    docker_ui_fail "Loss-classifier" "model_loaded=false (seed esperado)"
    exit 1
  fi
  if ! printf '%s' "${payload}" | grep -q '"veto_ready"[[:space:]]*:[[:space:]]*true'; then
    docker_ui_fail "Loss-classifier" "veto_ready=false (seed n_train>=READY_N esperado)"
    exit 1
  fi
  docker_ui_ok "ready=true model_loaded=true veto_ready=true"
}

case "${MODE}" in
  clear)
    docker_ui_banner "loss-models clear · limpa bind-mount (chamado por docker-rebuild/reset)"
    docker_ui_step 1 2 "Parar ${SERVICE}"
    stop_loss_if_running
    docker_ui_step 2 2 "Limpeza de ${MODELS_DIR}"
    clear_models
    docker_ui_done "loss-models limpo; caller sobe a stack"
    ;;
  full)
    docker_ui_banner "loss-models reset · limpa + restart loss (seed predictivo, floor FLIP 0.90)"
    docker_ui_step 1 4 "Parar ${SERVICE}"
    stop_loss_if_running
    docker_ui_step 2 4 "Limpeza de ${MODELS_DIR}"
    clear_models
    docker_ui_step 3 4 "Subir ${SERVICE}"
    "${COMPOSE[@]}" up -d "${SERVICE}"
    bash "${SCRIPT_DIR}/docker-wait-healthy.sh"
    docker_ui_ok "healthy"
    docker_ui_step 4 4 "Validar /health seed"
    validate_health_seeded
    docker_ui_done "loss-models semeado; reinicie o motor"
    ;;
  *)
    echo "uso: $0 [clear|full]" >&2
    exit 2
    ;;
esac
exit 0
