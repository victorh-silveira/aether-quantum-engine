# Aether Quantum Engine — Laya System 1 Continuo (`R_10` M5)

Motor autonomo de trading quantitativo na Deriv operando contratos Rise/Fall de **5 minutos** no indice sintetico **`R_10`** de forma **100% continua (zero SKIP)**, com toda a decisao binaria (`CALL` ou `PUT`) delegada ao modelo **Laya System 1** (`laya==0.5.5`).

---

## Arquitetura e Fluxo de Decisao

1. **Coleta M5 (`StreamHandler`):** consome velas M5 fechadas de `R_10` via WebSocket Deriv (e persiste ticks/velas no TimescaleDB).
2. **Indicadores Causais (`src/domain/indicators/technical_indicators.py`):** calcula `EMA(9,20,21,50)`, `RSI(7,14)`, `MACD`, `Stochastic`, `CCI(14)`, `CMO(14)`, `Williams%R`, `ROC`, `ADX/DI(14,20)`, `ATR(14,50)`, `Bollinger %B`, `Keltner %B`, `Hurst(14)`, `Z-Score` de sessao e microestrutura.
3. **Decisao Laya (`src/infrastructure/inference/laya_decision_adapter.py`):** envia `R_10`, ultimas velas OHLC fechadas, todos os indicadores e configuracoes sanitizadas de `config/settings.json` para `laya.load("convaiinnovations/laya", vegaml=False).system_one(...)`, retornando sempre `CALL` ou `PUT` com probabilidades calibradas.
4. **Stake Dinamica (`src/domain/risk/laya_stake.py`):** dimensiona a entrada linearmente entre **1% e 5%** da banca conforme a probabilidade do Laya (sem recuperacao 1/1 ou Martingale).
5. **Execucao e Liquidacao (`Orchestrator` + Redis ZSET):** compra Rise/Fall `5m` via `TradeHandler`, registra logs completos (`INDICATORS | ...` e `LAYA_DECISION | ...`) e reconcilia liquidacao via fila Redis ZSET.

---

## Comandos Principais (`Makefile` no WSL)

- `make docker-up`: sobe a stack de infraestrutura (`TimescaleDB`, `Redis`, `MinIO`, `Prometheus`, `Grafana`, `Tempo`).
- `make app-install`: instala dependencias de runtime (`laya==0.5.5`, `torch`, `polars`, `websockets`, `redis`, `asyncpg`) no Conda `deriv-api`.
- `make app-run`: executa o motor continuo no `R_10` M5 exibindo indicadores, direcao e probabilidades do Laya.
- `make app-test`: executa toda a suite de testes com **100% de cobertura**.
- `make app-lint` / `make app-security` / `make app-validate`: valida Ruff, Interrogate, Vulture, limite de 300 linhas/arquivo, Bandit e schemas JSON/YAML/Docker.
- `make app-commit ARGS="-m '...'"`: commit padronizado em PT-BR com hooks pre-commit.
