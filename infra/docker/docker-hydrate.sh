#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/docker-ui.sh"

if [ ! -f "${REPO_ROOT}/infra/docker/docker-compose.yml" ]; then
  echo "docker-hydrate: execute a partir da raiz do repositorio" >&2
  exit 1
fi

cd "${REPO_ROOT}"

if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

COMPOSE=(docker compose -f infra/docker/docker-compose.yml --project-directory infra/docker --env-file .env)
PG_USER="${AETHER_PG_USER:-aether}"
PG_DB="${AETHER_PG_DB:-aether}"

printf '  %sHydrate TimescaleDB%s\n' "${DOCKER_UI_BOLD}" "${DOCKER_UI_RESET}"
docker_ui_nl

CURRENT_COUNT="$("${COMPOSE[@]}" exec -T timescaledb psql -U "$PG_USER" -d "$PG_DB" -t -A -c "SELECT count(*) FROM ohlc_bars;" 2>/dev/null || echo "0")"
CURRENT_COUNT="$(echo "$CURRENT_COUNT" | tr -d '[:space:]')"
if [ -z "$CURRENT_COUNT" ]; then
  CURRENT_COUNT=0
fi

if [ "$CURRENT_COUNT" -lt 48 ]; then
  docker_ui_warn "fome de dados (${CURRENT_COUNT} barras) - hidratando lookback M1"
  "${COMPOSE[@]}" exec -T timescaledb psql -q -U "$PG_USER" -d "$PG_DB" -c "
    INSERT INTO ohlc_bars (time, symbol, epoch, granularity, open, high, low, close)
    SELECT t, sym, EXTRACT(EPOCH FROM t)::bigint, 60, 100.0+(i*0.01), 100.5+(i*0.01), 99.5+(i*0.01), 100.1+(i*0.01)
    FROM (SELECT NOW() - (i * INTERVAL '1 minute') AS t, i FROM generate_series(1, 60) i) s
    CROSS JOIN (SELECT 'R_10' AS sym) symbols
    ON CONFLICT DO NOTHING;" >/dev/null
  docker_ui_ok "lookback reidratado"
else
  docker_ui_ok "ohlc_bars (${CURRENT_COUNT} registros)"
fi

docker_ui_nl
