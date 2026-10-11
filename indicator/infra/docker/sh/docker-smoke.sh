#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"
compose=(docker compose -f infra/docker/docker-compose.yml)

for service in timescaledb indicator prometheus grafana; do
  if [[ -z "$("${compose[@]}" ps -q "$service")" ]]; then
    echo "FALHA: $service nao esta em execucao" >&2
    exit 1
  fi
done

for attempt in {1..30}; do
  if "${compose[@]}" exec -T timescaledb pg_isready -U aether -d aether >/dev/null 2>&1 \
    && curl -fsS http://127.0.0.1:9102/metrics >/dev/null \
    && curl -fsS http://127.0.0.1:19091/-/healthy >/dev/null \
    && curl -fsS -u admin:aether http://127.0.0.1:3001/api/datasources/uid/timescale/health | grep -q 'Database Connection OK'; then
    echo 'OK: Timescale, indicador, Prometheus e Grafana'
    exit 0
  fi
  if (( attempt == 30 )); then
    echo 'FALHA: a stack nao passou no smoke check em 60 segundos' >&2
    exit 1
  fi
  sleep 2
done
