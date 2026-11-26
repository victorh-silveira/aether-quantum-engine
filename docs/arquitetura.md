# Arquitetura do Aether Quantum Engine (Laya Continuo R_10 M5)

O motor segue **DDD (Domain-Driven Design)** e **Arquitetura Hexagonal (Ports & Adapters)** em **Python 3.13** com **asyncio** no host e stack de dados/telemetria em Docker (`Redis`, `TimescaleDB`, `MinIO`, `Prometheus`, `Grafana`, `Tempo`).

## Camadas DDD

- **Domain (`src/domain/`)**: 100% puro, sem imports de infraestrutura ou ML (`laya`, `torch`, `redis`, `asyncpg`).
  - `indicators/`: calculo causal de todos os indicadores tecnicos (`compute_indicators`, `IndicatorSnapshot`).
  - `models/`: `Candle`, `TradeDirection`, `Contract`, `LayaDecision` (estritamente `CALL` ou `PUT`).
  - `risk/`: `RiskManager`, `StakeSizing` e `size_kelly_recovery` (Kelly fracionado com piso de 1%, recuperacao por payout e teto absoluto de 5%; estado de liquidacao persistente e idempotente).
- **Application (`src/application/`)**:
  - `ports/outbound.py`: contratos `MarketCandlePort`, `SettlementQueuePort`, `LayaDecisionPort`.
  - `services/orchestrator/`: ciclo M5 continuo no `R_10`, warmup de velas fechadas, log obrigatorio `INDICATORS | ...` e `LAYA_DECISION | ...`, envio de ordens Rise/Fall 5m (`duration=5`, `duration_unit="m"`) e reconciliacao de liquidacao.
- **Infrastructure (`src/infrastructure/`)**:
  - `inference/laya_decision_adapter.py`: adapter de `LayaDecisionPort` usando `laya==0.5.5` (`convaiinnovations/laya`, `vegaml=False`, `agent.system_one(...)`) com sanitizacao de credenciais do `settings.json` e desempate deterministico (`ema_20 >= ema_50 -> CALL`).
  - `api/` e `handlers/`: WebSocket/REST Deriv, stream M5 (`R_10`), buffer de ticks e execucao de propostas.
  - `state/`, `market/`, `telemetry/`: Redis ZSET para liquidacao, TimescaleDB para contratos/velas e Prometheus/OTel.
- **Presentation (`src/presentation/`)**: logging estruturado em terminal e arquivo sem emojis.
