---
name: aether-asyncpg-timescale
description: Audita pool asyncpg, transacoes, ingestao e hypertables TimescaleDB no Aether.
---

# Ingestão e Persistência TimescaleDB via asyncpg

## 1. Contexto e Pré-requisitos
- Leia `docs/infra-docker.md` e `docs/engineering-devops-cloudops-senior.md`.
- Verifique que o container `aether-core-data-timescaledb` esteja saudavel (`make docker-status`).

## 2. Procedimento Operacional Passo a Passo
1. **Auditoria do Pool de Conexões:**
   - Assegure que o pool `asyncpg.create_pool` utilize configuracao fail-closed com `min_size` e `max_size` proporcionais a carga do host.
   - Valide se conexoes ociosas sao recicladas adequadamente.
2. **Consultas Preparadas e Hypertables:**
   - Verifique se insercoes de ticks e OHLC utilizam queries parametrizadas ou operacoes em lote `copy_records_to_table`.
   - Confirme se as tabelas temporais possuem particionamento nativo por chunk de tempo (hypertables do Timescale).
3. **Resiliência a Quedas:**
   - Implemente circuit breakers defensivos: se o banco falhar, o motor de cotacao deve continuar operando em memoria sem interrupcao do loop principal.

## 3. Comandos de Verificação
```bash
/mnt/c/Users/victor-silveira/anaconda3/envs/deriv-api/python.exe scripts/operations/ensure_timescale.py
/mnt/c/Users/victor-silveira/anaconda3/envs/deriv-api/python.exe -m pytest app/tests/unit/infrastructure/test_timescale*.py -v
```

## 4. Critérios de Aceite
- Conexoes pooladas e fechadas graciosamente.
- Consultas otimizadas com explain analisado para hypertables.
