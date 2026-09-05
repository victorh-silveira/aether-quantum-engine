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
SYMBOL="${AETHER_HYDRATE_SYMBOL:-1HZ75V}"
MICRO_G="${AETHER_HYDRATE_MICRO_GRANULARITY:-300}"
MACRO_G="${AETHER_HYDRATE_MACRO_GRANULARITY:-86400}"
MICRO_BARS="${AETHER_HYDRATE_MICRO_BARS:-500}"
MACRO_BARS="${AETHER_HYDRATE_MACRO_BARS:-365}"
MICRO_MIN="${AETHER_HYDRATE_MICRO_MIN:-400}"
MACRO_MIN="${AETHER_HYDRATE_MACRO_MIN:-200}"

printf '  %sHydrate TimescaleDB%s\n' "${DOCKER_UI_BOLD}" "${DOCKER_UI_RESET}"
docker_ui_nl

if ! "${COMPOSE[@]}" ps -q timescaledb 2>/dev/null | grep -q .; then
  docker_ui_warn "timescaledb inativo - hydrate ignorado"
  docker_ui_nl
  exit 0
fi

MICRO_COUNT="$("${COMPOSE[@]}" exec -T timescaledb psql -U "$PG_USER" -d "$PG_DB" -t -A -c "SELECT count(*) FROM ohlc_bars WHERE symbol='${SYMBOL}' AND granularity=${MICRO_G};" 2>/dev/null || echo "0")"
MICRO_COUNT="$(echo "$MICRO_COUNT" | tr -d '[:space:]')"
MACRO_COUNT="$("${COMPOSE[@]}" exec -T timescaledb psql -U "$PG_USER" -d "$PG_DB" -t -A -c "SELECT count(*) FROM ohlc_bars WHERE symbol='${SYMBOL}' AND granularity=${MACRO_G};" 2>/dev/null || echo "0")"
MACRO_COUNT="$(echo "$MACRO_COUNT" | tr -d '[:space:]')"
if [ -z "$MICRO_COUNT" ]; then MICRO_COUNT=0; fi
if [ -z "$MACRO_COUNT" ]; then MACRO_COUNT=0; fi

if [ "$MICRO_COUNT" -lt "$MICRO_MIN" ] || [ "$MACRO_COUNT" -lt "$MACRO_MIN" ]; then
  docker_ui_warn "fome de dados (micro${MICRO_G}=${MICRO_COUNT} macro${MACRO_G}=${MACRO_COUNT}) - hidratando ${SYMBOL} M5 (${MICRO_G}s) e D1 (${MACRO_G}s)"
  "${COMPOSE[@]}" exec -T timescaledb psql -q -U "$PG_USER" -d "$PG_DB" -c "
    INSERT INTO ohlc_bars (time, symbol, epoch, granularity, open, high, low, close)
    SELECT t, '${SYMBOL}', EXTRACT(EPOCH FROM t)::bigint, ${MICRO_G},
           100.0+(i*0.01), 100.5+(i*0.01), 99.5+(i*0.01), 100.1+(i*0.01)
    FROM (SELECT NOW() - (i * INTERVAL '${MICRO_G} seconds') AS t, i FROM generate_series(1, ${MICRO_BARS}) i) s
    ON CONFLICT DO NOTHING;
    INSERT INTO ohlc_bars (time, symbol, epoch, granularity, open, high, low, close)
    SELECT t, '${SYMBOL}', EXTRACT(EPOCH FROM t)::bigint, ${MACRO_G},
           100.0+(i*0.02), 100.8+(i*0.02), 99.2+(i*0.02), 100.2+(i*0.02)
    FROM (SELECT NOW() - (i * INTERVAL '${MACRO_G} seconds') AS t, i FROM generate_series(1, ${MACRO_BARS}) i) s
    ON CONFLICT DO NOTHING;" >/dev/null
  docker_ui_ok "lookback ${SYMBOL} M5 (${MICRO_G}s x${MICRO_BARS}) e D1 (${MACRO_G}s x${MACRO_BARS}) reidratado"
else
  docker_ui_ok "ohlc_bars ${SYMBOL} pronto (micro${MICRO_G}=${MICRO_COUNT} macro${MACRO_G}=${MACRO_COUNT})"
fi

docker_ui_nl
