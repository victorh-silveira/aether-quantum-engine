#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/docker-ui.sh"

if [ ! -f "${REPO_ROOT}/infra/docker/docker-compose.yml" ]; then
  echo "docker-smoke: execute a partir da raiz do repositorio" >&2
  exit 1
fi

cd "${REPO_ROOT}"

if [ ! -f .env ]; then
  echo "docker-smoke: arquivo .env ausente na raiz" >&2
  exit 1
fi

COMPOSE=(docker compose -f infra/docker/docker-compose.yml --project-directory infra/docker --env-file .env)
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

if service_running redis; then
  checked=$((checked + 1))
  if ! "${COMPOSE[@]}" exec -T redis redis-cli ping 2>/dev/null | grep -qi PONG; then
    smoke_fail "Redis" "PING"
  else
    docker_ui_ok "Redis"
  fi
fi

if service_running timescaledb; then
  checked=$((checked + 1))
  if ! "${COMPOSE[@]}" exec -T timescaledb sh -c 'pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB"' >/dev/null 2>&1; then
    smoke_fail "TimescaleDB" "pg_isready"
  else
    docker_ui_ok "TimescaleDB"
  fi
fi

if service_running minio; then
  checked=$((checked + 1))
  if ! curl -sf "http://127.0.0.1:9000/minio/health/live" >/dev/null 2>&1; then
    smoke_fail "MinIO" "/minio/health/live"
  else
    docker_ui_ok "MinIO"
  fi
fi

if service_running aether-triton; then
  checked=$((checked + 1))
  if ! curl -sf "http://127.0.0.1:8000/v2/health/live" >/dev/null 2>&1; then
    smoke_fail "Triton" "/v2/health/live"
  else
    docker_ui_ok "Triton"
  fi
fi

if service_running aether-meta-classifier; then
  checked=$((checked + 1))
  if ! curl -sf "http://127.0.0.1:8005/health" >/dev/null 2>&1; then
    smoke_fail "Meta-classifier" "/health"
  else
    docker_ui_ok "Meta-classifier"
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
