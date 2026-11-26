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
- Kelly e recuperacao `[1%, 5%]`: `src/domain/risk/kelly_recovery.py`, com saldo e payout revalidados na proposta final.
- Adapter Laya: `src/infrastructure/inference/laya_decision_adapter.py`
- Contexto completo: `src/infrastructure/inference/laya_context.py`; conferir `LAYA_CONTEXT` com zero tokens descartados.
- Exibicao: indicadores, probabilidades, confianca, latencia e dinheiro com duas casas decimais; nao arredondar dados internos ou metricas Prometheus.
- Dados: `StreamHandler.get_closed_candles(symbol, 300)` exige M5 fechado, sem fallback diario; warmup minimo de 51 barras.
- Validacao offline: `python -m tools.laya_inference_probe` com pesos locais e sem executor de ordens. Nao exigir variacao artificial de probabilidades ou prometer rentabilidade.
- Orquestrador continuo: `src/application/services/orchestrator/trading_cycle_entry.py`

## Treinamento e Promocao

Usar `make app-prepare`, `make app-train`, `make app-evaluate` e `make app-promote`. Treino exige CUDA; `make app-train-dry-run` nao carrega pesos. Perfil auto=head na RTX 4060 de 8 GiB; full exige 12 GiB livres. Validar CUDA com `make app-cuda-check` e `make app-cuda-train-check`. Seguir divisao temporal 60/10/10/20 e criterios de `docs/laya-training.md`. Nao consumir teste para selecionar seeds ou ativar checkpoint sem `approval.json` integro.
