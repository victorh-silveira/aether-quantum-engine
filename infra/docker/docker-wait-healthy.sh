#!/usr/bin/env bash
set -euo pipefail

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
  for service in $(service_ids); do
    status="$(service_health "$service")"
    case "$status" in
      healthy|running)
        continue
        ;;
      starting)
        return 1
        ;;
      *)
        echo "docker-wait-healthy: servico $service em estado $status"
        return 2
        ;;
    esac
  done
  return 0
}

main() {
  local elapsed=0
  while [ "$elapsed" -lt "$TIMEOUT_SECS" ]; do
    if all_healthy; then
      "${COMPOSE[@]}" ps
      exit 0
    fi
    local code=$?
    if [ "$code" -eq 2 ]; then
      "${COMPOSE[@]}" ps
      exit 1
    fi
    sleep "$INTERVAL_SECS"
    elapsed=$((elapsed + INTERVAL_SECS))
  done
  echo "docker-wait-healthy: timeout apos ${TIMEOUT_SECS}s aguardando healthchecks"
  "${COMPOSE[@]}" ps
  exit 1
}

main "$@"
