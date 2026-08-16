#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/docker-ui.sh"
source "${SCRIPT_DIR}/compose-lib.sh"

if [ ! -f "${REPO_ROOT}/infra/docker/docker-compose.yml" ]; then
  echo "docker-wait-healthy: execute a partir da raiz do repositorio (compose nao encontrado)" >&2
  exit 1
fi

cd "${REPO_ROOT}"

if [ ! -f .env ]; then
  echo "docker-wait-healthy: arquivo .env ausente na raiz (cp .env.example .env)" >&2
  exit 1
fi

mapfile -t COMPOSE_FLAGS < <(compose_args)
COMPOSE=(docker compose "${COMPOSE_FLAGS[@]}")
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

print_status() {
  docker_ui_nl
  printf '  %sContainers%s\n' "${DOCKER_UI_BOLD}" "${DOCKER_UI_RESET}"
  docker_ui_nl
  "${COMPOSE[@]}" ps
  docker_ui_nl
}

main() {
  local elapsed=0
  local code=0
  printf '  %sAguardando healthchecks%s (timeout %ss | profiles=%s)\n' \
    "${DOCKER_UI_DIM}" "${DOCKER_UI_RESET}" "${TIMEOUT_SECS}" "${COMPOSE_PROFILES:-${DOCKER_PROFILES:-core,ml}}"
  while [ "$elapsed" -lt "$TIMEOUT_SECS" ]; do
    code=0
    all_healthy || code=$?
    if [ "$code" -eq 0 ]; then
      print_status
      exit 0
    fi
    if [ "$code" -eq 2 ]; then
      print_status
      exit 1
    fi
    sleep "$INTERVAL_SECS"
    elapsed=$((elapsed + INTERVAL_SECS))
  done
  echo "docker-wait-healthy: timeout apos ${TIMEOUT_SECS}s aguardando healthchecks" >&2
  print_status
  exit 1
}

main "$@"
