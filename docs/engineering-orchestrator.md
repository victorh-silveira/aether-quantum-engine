# Orchestrator e Ciclo Continuo M5 (`R_10`)

O orquestrador (`src/application/services/orchestrator/`) opera em loop continuo no `R_10` com velas M5 fechadas (`period_seconds=300`), sem `make app-train`, sem TCN e sem vetos ou `SKIP`.

## Fluxo de Cada Ciclo M5

1. **Liquidacao pendente:** `process_redis_settlement_queue(orch)` verifica contratos vencidos na fila ZSET.
2. **Guarda de cadencia (`trading_cycle_entry_guards.py`):** garante execucao unica por fronteira M5 fechada (`m5_boundary_epoch`) quando nao ha contrato ativo nem hibernacao por manutencao da API.
3. **Warmup (`warm_up_buffer_guard.py`):** exige minimo de velas M5 fechadas no buffer para calculo causal estavel dos indicadores.
4. **Indicadores (`compute_indicators`):** calcula todo o pacote tecnico (EMAs, RSI, MACD, Stochastic, CCI, CMO, Williams %R, ROC, ADX/DI, ATR, Bollinger, Keltner, Hurst, microestrutura e `session_zscore`).
5. **Decisao Laya (`LayaDecisionAdapter.decide`):** envia velas recentes, indicadores e settings sanitizados ao Laya System 1, retornando `CALL` ou `PUT` (desempate deterministico por `ema_20 >= ema_50`).
6. **Dimensionamento (`calculate_laya_stake`):** calcula stake linear entre 1% e 5% da banca pela probabilidade do Laya, com piso `stake_min` e teto absoluto de 5%.
7. **Telemetria (`cycle_indicators_logger.py`):** imprime obrigatoriamente `INDICATORS | ...` e `LAYA_DECISION | ...`.
8. **Execucao:** `ExecutionManager.execute_laya_order` envia o contrato Rise/Fall de 5 minutos e registra o watch de liquidacao.
