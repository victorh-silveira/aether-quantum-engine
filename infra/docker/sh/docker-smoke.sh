#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${SCRIPT_DIR}/docker-ui.sh"
source "${SCRIPT_DIR}/compose-lib.sh"

if [ ! -f "${REPO_ROOT}/infra/docker/docker-compose.yml" ]; then
  echo "docker-smoke: execute a partir da raiz do repositorio" >&2
  exit 1
fi

cd "${REPO_ROOT}"

if [ ! -f .env ]; then
  echo "docker-smoke: arquivo .env ausente na raiz" >&2
  exit 1
fi

mapfile -t COMPOSE_FLAGS < <(compose_args)
COMPOSE=(docker compose "${COMPOSE_FLAGS[@]}")
fail=0
checked=0

smoke_fail() {
  docker_ui_fail "$1" "${2:-}"
  fail=1
}

service_running() {
  local service="$1"
  local id
  id="$("${COMPOSE[@]}" ps -q "$service" 2>/dev/null | head -n 1)"
  [ -n "$id" ]
}

require_service() {
  local service="$1"
  local label="$2"
  if ! service_running "$service"; then
    smoke_fail "$label" "profile ativo mas container parado"
    return 1
  fi
  return 0
}

if profile_active core; then
  require_service aether-core-cache-redis Redis || true
  if service_running aether-core-cache-redis; then
    checked=$((checked + 1))
    if ! "${COMPOSE[@]}" exec -T aether-core-cache-redis redis-cli ping 2>/dev/null | grep -qi PONG; then
      smoke_fail "Redis" "PING"
    else
      docker_ui_ok "Redis"
    fi
  fi

  require_service aether-core-data-timescaledb TimescaleDB || true
  if service_running aether-core-data-timescaledb; then
    checked=$((checked + 1))
    if ! "${COMPOSE[@]}" exec -T aether-core-data-timescaledb sh -c 'pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB"' >/dev/null 2>&1; then
      smoke_fail "TimescaleDB" "pg_isready"
    else
      docker_ui_ok "TimescaleDB"
    fi
  fi

  require_service aether-core-storage-minio MinIO || true
  if service_running aether-core-storage-minio; then
    checked=$((checked + 1))
    if ! curl -sf "http://127.0.0.1:9000/minio/health/live" >/dev/null 2>&1; then
      smoke_fail "MinIO" "/minio/health/live"
    else
      docker_ui_ok "MinIO"
    fi
  fi
fi

if profile_active ml; then
  require_service aether-ml-meta-classifier Meta-classifier || true
  if service_running aether-ml-meta-classifier; then
    checked=$((checked + 1))
    meta_payload="$(curl -sf "http://127.0.0.1:8005/health" 2>/dev/null || true)"
    if [ -z "$meta_payload" ]; then
      smoke_fail "Meta-classifier" "/health"
    elif ! printf '%s' "$meta_payload" | grep -q '"ready"[[:space:]]*:[[:space:]]*true'; then
      docker_ui_warn "Meta-classifier ready=false (rode launch-train → meta-models/meta_lgbm.pkl)"
      docker_ui_ok "Meta-classifier (processo up; modelo ausente ate train)"
    else
      docker_ui_ok "Meta-classifier"
    fi
  fi
  require_service aether-ml-loss-classifier Loss-classifier || true
  if service_running aether-ml-loss-classifier; then
    checked=$((checked + 1))
    loss_payload="$(curl -sf "http://127.0.0.1:8006/health" 2>/dev/null || true)"
    if [ -z "$loss_payload" ]; then
      smoke_fail "Loss-classifier" "/health"
    elif ! printf '%s' "$loss_payload" | grep -q '"ready"[[:space:]]*:[[:space:]]*true'; then
      smoke_fail "Loss-classifier" "ready=false"
    else
      docker_ui_ok "Loss-classifier"
    fi
  fi
fi

if profile_active indicator; then
  require_service aether-market-signal-indicator Indicador || true
  if service_running aether-market-signal-indicator; then
    checked=$((checked + 1))
    if ! curl -sf "http://127.0.0.1:9102/metrics" >/dev/null 2>&1; then
      smoke_fail "Indicador" "/metrics"
    else
      docker_ui_ok "Indicador"
    fi
  fi
fi

if profile_active telemetry; then
  require_service aether-observe-metrics-prometheus Prometheus || true
  if service_running aether-observe-metrics-prometheus; then
    checked=$((checked + 1))
    if ! curl -sf "http://127.0.0.1:9090/-/healthy" >/dev/null 2>&1; then
      smoke_fail "Prometheus" "/-/healthy"
    else
      docker_ui_ok "Prometheus"
    fi
  fi

  require_service aether-observe-dashboards-grafana Grafana || true
  if service_running aether-observe-dashboards-grafana; then
    checked=$((checked + 1))
    if ! curl -sf "http://127.0.0.1:3000/api/health" >/dev/null 2>&1; then
      smoke_fail "Grafana" "/api/health"
    else
      docker_ui_ok "Grafana"
    fi
  fi
fi

docker_ui_nl

if [ "$fail" -ne 0 ]; then
  docker_ui_warn "um ou mais checks falharam"
  docker_ui_nl
  "${COMPOSE[@]}" ps
  docker_ui_nl
  exit 1
fi

if [ "$checked" -eq 0 ]; then
  docker_ui_warn "nenhum servico ativo para validar"
  docker_ui_nl
  exit 1
fi

docker_ui_done "docker-smoke: stack OK (${checked} checks)"
exit 0
