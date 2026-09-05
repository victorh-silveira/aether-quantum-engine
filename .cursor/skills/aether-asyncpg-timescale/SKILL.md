---
name: aether-asyncpg-timescale
description: >-
  Audita acesso asyncpg/TimescaleDB no Aether (pool fail-closed, prepared,
  COPY/lote, hypertables). Use when changing DB adapters, insert paths,
  pool tuning, or Timescale schema/retention.
---

# asyncpg / TimescaleDB

## Quando aplicar

Adapter de persistencia, inserts de ticks/velas, tuning de pool, hypertables/compressao/retencao, ou latencia de gravação acoplada ao ciclo.

## Checklist

1. Ler `docs/engineering-python-313-runtime.md` + `docs/infra-docker.md` + `docs/engineering-devops-cloudops-senior.md` + `docs/engineering-architecture-senior.md`
2. Cliente: **asyncpg** (async); sem driver sync no hot path do motor
3. Pool transacional **fail-closed**; timeouts alinhados ao compose/healthcheck
4. Preferir **prepared statements** e inserts em lote / `COPY` quando o volume justificar
5. Gravacao **desacoplada** da decisao CALL/PUT (nao bloquear gates por flush DB)
6. Hypertables: chunk **1 day**, `compress_segmentby`/`compress_orderby`, politicas de retencao; CRAG `candle_m5` so analytics
7. Bind Docker Timescale em `127.0.0.1`; segredos so env/secret store
8. Domain puro: sem import asyncpg em `domain/`

## Anti-padroes

- `psycopg2`/`sqlite3` sync no loop asyncio do motor
- Um INSERT por tick sem batch sob carga
- Segurar o lock do ciclo esperando commit Timescale
- Subir `max_connections` sem alinhar pool asyncpg + cgroup do container
- Colocar SQL/I/O dentro de entidades de dominio
- Usar CRAG como substituto do candle live Deriv no orquestrador

## Refs no repo

- `docs/engineering-python-313-runtime.md`
- `docs/engineering-devops-cloudops-senior.md`
- `docs/infra-docker.md`
- `docs/engineering-architecture-senior.md`
- `infra/docker/003_init-timescale.sql` / `005_timescale_crags.sql`
- `app/src/infrastructure/` (adapters asyncpg/Timescale)
- `.cursor/rules/aether-infra.mdc`
- `.cursor/rules/aether-python-313-runtime.mdc`
- `.cursor/rules/aether-domain-pure.mdc`

## Skills irmas

`aether-infra-stack`, `aether-devops-cloudops`, `aether-python-313-runtime`, `aether-asyncio-supervisor`, `aether-architecture-senior`, `aether-settlement-debug`
