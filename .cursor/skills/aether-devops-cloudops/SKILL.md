---
name: aether-devops-cloudops
description: >-
  Audita e endurece DevOps/CloudOps Aether (Compose, Redis, Timescale, MinIO,
  multi-stage meta/loss, health/depends_on, OOM/OMP). Use when changing
  docker-compose, redis.conf, Timescale SQL, MinIO bootstrap, or sidecar images.
---

# DevOps & CloudOps sênior

## Quando aplicar

Mudanca em `infra/docker/**`, healthchecks, cgroups, AOF/maxmemory, hypertables/CRAGs, bootstrap MinIO/ILM, ou Dockerfiles meta/loss.

## Checklist

1. Ler `docs/engineering-devops-cloudops-senior.md` + `docs/infra-docker.md`
2. Binds só `127.0.0.1`; processo interno em `0.0.0.0`
3. `mem_limit`/`cpus` + `mem_swappiness: 0`; nao depender de Swarm `deploy.resources`
4. Meta/loss: multi-stage `/opt/venv`, `tini`, CMD exec-form, OMP/OpenBLAS/MKL/VECLIB/NUMEXPR = 2
5. Redis: AOF everysec, `maxmemory` + `noeviction`, io-threads; settlement = ZSET (nunca Streams sem mandato)
6. Timescale: chunk 1d, compressao/retencao, CRAG `candle_m5` so analytics
7. MinIO: bucket `dl-models`, `minio-init` + ILM optuna ~7d
8. `depends_on: service_healthy` + Make wait-healthy; host `vm.overcommit_memory=1`
9. Atualizar testes de contrato em `test_infra_docker.py` / `test_infra_redis_config.py`
10. Surface-sync: rule `aether-infra`, skills irmas, AGENTS/matriz

## Anti-padroes

- Eviction LRU no Redis de settlement
- Migrar `settlement:queue:priority` para Streams/listas
- Shell-form CMD no sidecar Python
- Expor portas na interface externa
- Rewire candle live para CRAG

## Refs

- `docs/engineering-devops-cloudops-senior.md`
- `docs/infra-docker.md`
- `docs/engineering-settlement.md`
- `infra/docker/docker-compose.yml`
- `.cursor/rules/aether-infra.mdc`

## Skills irmas

`aether-infra-stack`, `aether-redis-hiredis`, `aether-asyncpg-timescale`, `aether-settlement-debug`, `aether-surface-sync`
