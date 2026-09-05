---
name: aether-redis-hiredis
description: >-
  Audita Redis Aether com hiredis e fila ZSET settlement:queue:priority.
  Use when changing Redis clients, settlement enqueue/consume, AOF, or
  proposals to replace ZSET with Streams.
---

# Redis + hiredis / settlement ZSET

## Quando aplicar

Cliente Redis, fila de liquidacao, orphans/estagnacao SETTLE, mudanca AOF, ou proposta de trocar a estrutura da fila.

## Checklist

1. Ler `docs/engineering-python-313-runtime.md` + `docs/engineering-settlement.md` + `docs/infra-docker.md`
2. Parser C: cliente com **hiredis** onde o stack do motor permitir
3. Settlement SSOT: Redis **ZSET** `settlement:queue:priority` — enqueue idempotente por `contract_id`
4. **Nunca** substituir por Streams, listas ou outro tipo sem mandato explicito + migracao testada
5. Pipeline/transacao atomica para state stores; isolamento com `asyncio.Lock` no core quando necessario
6. Writes criticos: considerar `asyncio.shield` sob cancel do ciclo (skill `aether-asyncio-supervisor`)
7. AOF `appendfsync everysec`; `maxmemory` + `noeviction`; `io-threads`; bind `127.0.0.1:6379`; nao apagar fila sem auditar contratos abertos
8. Tolerancia settle ~600 s; poll minimo ~2 s; preservar spam filter / dedupe de logs SETTLE
9. Doutrina CloudOps: `docs/engineering-devops-cloudops-senior.md`

## Anti-padroes

- Migrar settlement para Redis Streams/listas “por performance” sem mandato
- Apagar `settlement:queue:priority` em prod para “limpar ruido”
- Cliente sync bloqueando o loop
- Expor Redis fora de loopback
- Duplicar enqueue sem idempotencia por `contract_id`
- Eviction LRU (`volatile-lru` / `allkeys-lru`) na instancia de settlement

## Refs no repo

- `docs/engineering-settlement.md`
- `docs/engineering-devops-cloudops-senior.md`
- `docs/engineering-python-313-runtime.md`
- `docs/infra-docker.md`
- `docs/engineering-architecture-senior.md`
- `infra/docker/redis.conf`
- `app/src/` (ops de `settlement:queue:priority` / Redis adapters)
- `.cursor/rules/aether-settlement.mdc`
- `.cursor/rules/aether-infra.mdc`
- `.cursor/rules/aether-python-313-runtime.mdc`

## Skills irmas

`aether-settlement-debug`, `aether-asyncio-supervisor`, `aether-infra-stack`, `aether-devops-cloudops`, `aether-python-313-runtime`, `aether-architecture-senior`
