---
name: aether-laya-engine
description: "Execucao, diagnostico e manutencao do motor continuo Laya System 1 no R_10 M5."
---

# Skill: Aether Laya Engine (`R_10` M5 Continuo)

## Fluxo Operacional
1. Subir infraestrutura Docker: `make docker-up` (Redis, TimescaleDB, MinIO, Prometheus, Grafana, Tempo).
2. Executar motor continuo: `make app-run` (Conda `deriv-api`, Python 3.13, `laya==0.5.5`).
3. Verificar logs em `logs/engine.log` contendo `INDICATORS | R_10 M5 ...` e `LAYA_DECISION | R_10 M5 dir=CALL|PUT ...`.

## Manutencao
- Indicadores: `src/domain/indicators/technical_indicators.py`
- Modelo de decisao: `src/domain/models/laya_decision.py`
- Stake dinamica `[1%, 5%]`: `src/domain/risk/laya_stake.py`
- Adapter Laya: `src/infrastructure/inference/laya_decision_adapter.py`
- Orquestrador continuo: `src/application/services/orchestrator/trading_cycle_entry.py`
