#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

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

smoke_fail() {
  echo "docker-smoke: FALHA - $1" >&2
  fail=1
}

service_running() {
  local service="$1"
  local id
  id="$("${COMPOSE[@]}" ps -q "$service" 2>/dev/null | head -n 1)"
  [ -n "$id" ]
}

if service_running redis; then
  if ! "${COMPOSE[@]}" exec -T redis redis-cli ping 2>/dev/null | grep -qi PONG; then
    smoke_fail "Redis PING"
  else
    echo "docker-smoke: Redis OK"
  fi
fi

if service_running timescaledb; then
  if ! "${COMPOSE[@]}" exec -T timescaledb sh -c 'pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB"' >/dev/null 2>&1; then
    smoke_fail "TimescaleDB pg_isready"
  else
    echo "docker-smoke: TimescaleDB OK"
  fi
fi

if service_running minio; then
  if ! curl -sf "http://127.0.0.1:9000/minio/health/live" >/dev/null 2>&1; then
    smoke_fail "MinIO /minio/health/live"
  else
    echo "docker-smoke: MinIO OK"
  fi
fi

if service_running aether-triton; then
  if ! curl -sf "http://127.0.0.1:8000/v2/health/live" >/dev/null 2>&1; then
    smoke_fail "Triton /v2/health/live"
  else
    echo "docker-smoke: Triton OK"
  fi
fi

if service_running aether-meta-classifier; then
  if ! curl -sf "http://127.0.0.1:8005/health" >/dev/null 2>&1; then
    smoke_fail "Meta /health"
  else
    echo "docker-smoke: Meta-classifier OK"
  fi
fi

if [ "$fail" -ne 0 ]; then
  "${COMPOSE[@]}" ps
  exit 1
fi

echo "docker-smoke: stack OK"
exit 0
