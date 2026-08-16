#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/docker-ui.sh"
source "${SCRIPT_DIR}/compose-lib.sh"

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

mapfile -t COMPOSE_FLAGS < <(compose_args)
COMPOSE=(docker compose "${COMPOSE_FLAGS[@]}")
PG_USER="${AETHER_PG_USER:-aether}"
PG_DB="${AETHER_PG_DB:-aether}"

printf '  %sHydrate TimescaleDB%s\n' "${DOCKER_UI_BOLD}" "${DOCKER_UI_RESET}"
docker_ui_nl

if ! "${COMPOSE[@]}" ps -q timescaledb 2>/dev/null | grep -q .; then
  docker_ui_warn "timescaledb inativo - hydrate ignorado"
  docker_ui_nl
  exit 0
fi

MICRO_COUNT="$("${COMPOSE[@]}" exec -T timescaledb psql -U "$PG_USER" -d "$PG_DB" -t -A -c "SELECT count(*) FROM ohlc_bars WHERE symbol='R_10' AND granularity=60;" 2>/dev/null || echo "0")"
MICRO_COUNT="$(echo "$MICRO_COUNT" | tr -d '[:space:]')"
MACRO_COUNT="$("${COMPOSE[@]}" exec -T timescaledb psql -U "$PG_USER" -d "$PG_DB" -t -A -c "SELECT count(*) FROM ohlc_bars WHERE symbol='R_10' AND granularity=7200;" 2>/dev/null || echo "0")"
MACRO_COUNT="$(echo "$MACRO_COUNT" | tr -d '[:space:]')"
if [ -z "$MICRO_COUNT" ]; then MICRO_COUNT=0; fi
if [ -z "$MACRO_COUNT" ]; then MACRO_COUNT=0; fi

if [ "$MICRO_COUNT" -lt 720 ] || [ "$MACRO_COUNT" -lt 80 ]; then
  docker_ui_warn "fome de dados (micro60=${MICRO_COUNT} macro7200=${MACRO_COUNT}) - hidratando R_10 M1"
  "${COMPOSE[@]}" exec -T timescaledb psql -q -U "$PG_USER" -d "$PG_DB" -c "
    INSERT INTO ohlc_bars (time, symbol, epoch, granularity, open, high, low, close)
    SELECT t, 'R_10', EXTRACT(EPOCH FROM t)::bigint, 60,
           100.0+(i*0.01), 100.5+(i*0.01), 99.5+(i*0.01), 100.1+(i*0.01)
    FROM (SELECT NOW() - (i * INTERVAL '60 seconds') AS t, i FROM generate_series(1, 1333) i) s
    ON CONFLICT DO NOTHING;
    INSERT INTO ohlc_bars (time, symbol, epoch, granularity, open, high, low, close)
    SELECT t, 'R_10', EXTRACT(EPOCH FROM t)::bigint, 7200,
           100.0+(i*0.02), 100.8+(i*0.02), 99.2+(i*0.02), 100.2+(i*0.02)
    FROM (SELECT NOW() - (i * INTERVAL '7200 seconds') AS t, i FROM generate_series(1, 160) i) s
    ON CONFLICT DO NOTHING;" >/dev/null
  docker_ui_ok "lookback macro/micro M1 reidratado"
else
  docker_ui_ok "ohlc_bars R_10 (micro60=${MICRO_COUNT} macro7200=${MACRO_COUNT})"
fi

docker_ui_nl
