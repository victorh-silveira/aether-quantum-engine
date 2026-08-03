#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/docker-ui.sh"
source "${SCRIPT_DIR}/compose-lib.sh"

if [ ! -f "${REPO_ROOT}/infra/docker/docker-compose.yml" ]; then
  echo "timescale-lifecycle: execute a partir da raiz do repositorio" >&2
  exit 1
fi

cd "${REPO_ROOT}"

if [ ! -f .env ]; then
  echo "timescale-lifecycle: arquivo .env ausente na raiz" >&2
  exit 1
fi

mapfile -t COMPOSE_FLAGS < <(compose_args)
COMPOSE=(docker compose "${COMPOSE_FLAGS[@]}")

printf '  %sTimescale lifecycle%s\n' "${DOCKER_UI_BOLD}" "${DOCKER_UI_RESET}"
docker_ui_nl

"${COMPOSE[@]}" up -d timescaledb

ready=0
for _ in $(seq 1 40); do
  if "${COMPOSE[@]}" exec -T timescaledb sh -c 'pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB"' >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 1
done

if [ "$ready" -ne 1 ]; then
  docker_ui_fail "TimescaleDB" "pg_isready timeout"
  exit 1
fi

"${COMPOSE[@]}" exec -T -e PGOPTIONS='-c client_min_messages=warning' timescaledb sh -c \
  'psql -q -U "$POSTGRES_USER" -d "$POSTGRES_DB" -v ON_ERROR_STOP=1 -f /docker-scripts/004_timescale-lifecycle.sql >/dev/null'

docker_ui_ok "compressao/retencao"
docker_ui_nl
