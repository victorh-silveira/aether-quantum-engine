#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ ! -f "${REPO_ROOT}/infra/docker/docker-compose.yml" ]; then
  echo "docker-wait-healthy: execute a partir da raiz do repositorio (compose nao encontrado)" >&2
  exit 1
fi

cd "${REPO_ROOT}"

if [ ! -f .env ]; then
  echo "docker-wait-healthy: arquivo .env ausente na raiz (cp .env.example .env)" >&2
  exit 1
fi

COMPOSE=(docker compose -f infra/docker/docker-compose.yml --project-directory infra/docker --env-file .env)
TIMEOUT_SECS="${AETHER_DOCKER_HEALTH_TIMEOUT:-300}"
INTERVAL_SECS=3

service_ids() {
  "${COMPOSE[@]}" ps --services 2>/dev/null
}

service_health() {
  local service="$1"
  local container
  container="$("${COMPOSE[@]}" ps -q "$service" 2>/dev/null | head -n 1)"
  if [ -z "$container" ]; then
    echo "missing"
    return 0
  fi
  docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$container" 2>/dev/null || echo "missing"
}

all_healthy() {
  local service status
  local ids
  ids="$(service_ids)"
  if [ -z "${ids}" ]; then
    echo "docker-wait-healthy: nenhum servico ativo para os profiles atuais" >&2
    return 2
  fi
  for service in ${ids}; do
    status="$(service_health "$service")"
    case "$status" in
      healthy)
        continue
        ;;
      starting)
        return 1
        ;;
      *)
        echo "docker-wait-healthy: servico $service em estado $status" >&2
        return 2
        ;;
    esac
  done
  return 0
}

main() {
  local elapsed=0
  local code=0
  while [ "$elapsed" -lt "$TIMEOUT_SECS" ]; do
    code=0
    all_healthy || code=$?
    if [ "$code" -eq 0 ]; then
      "${COMPOSE[@]}" ps
      exit 0
    fi
    if [ "$code" -eq 2 ]; then
      "${COMPOSE[@]}" ps
      exit 1
    fi
    sleep "$INTERVAL_SECS"
    elapsed=$((elapsed + INTERVAL_SECS))
  done
  echo "docker-wait-healthy: timeout apos ${TIMEOUT_SECS}s aguardando healthchecks" >&2
  "${COMPOSE[@]}" ps
  exit 1
}

main "$@"
