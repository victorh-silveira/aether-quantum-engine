---
name: aether-redis-hiredis
description: Auditoria de conexoes Redis com parser hiredis, persistencia AOF e filas ZSET.
---

# Gerenciamento Redis com hiredis e Filas ZSET

## 1. Contexto e Pré-requisitos
- Leia `docs/engineering-settlement.md` e `.agents/rules/aether-settlement.mdc`.

## 2. Procedimento Operacional Passo a Passo
1. **Auditoria de Conexão:**
   - Assegure que o cliente `redis-py` esteja utilizando o parser C de alta velocidade `hiredis`.
2. **Operações Atômicas de Fila:**
   - Enfileiramento de contratos em resolucao deve utilizar estritamente o ZSET `settlement:queue:priority` com timestamps de expiracao como score.
   - Remocoes e consultas de contratos vencidos devem ocorrer via scripts Lua ou comandos atomicos (`ZPOPMIN`, `ZRANGEBYSCORE`).
3. **Auditoria de Persistência:**
   - Confirme no `redis.conf` que `appendonly yes` e `appendfsync everysec` estejam ativos.

## 3. Comandos de Verificação
```bash
/mnt/c/Users/victor-silveira/anaconda3/envs/deriv-api/python.exe -m pytest app/tests/unit/infrastructure/test_redis*.py -v
```

## 4. Critérios de Aceite
- Conexoes Redis operando com hiredis.
- Zero perda de mensagens ou contratos orfaos na fila.
