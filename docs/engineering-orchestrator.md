# Orchestrator e Ciclo Continuo M5 (`R_10`)

O orquestrador (`src/application/services/orchestrator/`) opera em loop continuo no `R_10` com velas M5 fechadas (`period_seconds=300`) e filtros `SKIP` opcionais configurados exclusivamente via `laya.skip_policy` em `config/settings.json`. `make app-run` verifica CUDA e inicia o motor; `make app-train` executa o treinamento offline separado, sem enviar ordens.

## Fluxo de Cada Ciclo M5

1. **Liquidacao pendente:** `process_redis_settlement_queue(orch)` verifica contratos vencidos na fila ZSET.
2. **Guarda de cadencia (`trading_cycle_entry_guards.py`):** garante execucao unica por fronteira M5 fechada (`m5_boundary_epoch`) quando nao ha contrato ativo nem hibernacao por manutencao da API.
3. **Warmup (`warm_up_buffer_guard.py`):** exige minimo de 51 velas M5 fechadas no buffer para calculo causal estavel dos indicadores.
4. **Indicadores (`compute_indicators`):** calcula o pacote tecnico completo (EMAs, Wilder RSI, MACD, Stochastic, CCI, CMO, Williams %R, ROC, ADX/DI, ATR, Bollinger %B/Width, Keltner %B, Hurst 50, microestrutura e `session_zscore`).
5. **Decisao Laya (`LayaDecisionAdapter.decide`):** envia narrativa tecnica em 3 estados, velas compactas e settings sanitizados ao Laya System 1 com inferencia simetrica, retornando `CALL` ou `PUT` (desempate deterministico por `ema_20 >= ema_50`).
6. **Avaliacao `SKIP` (`evaluate_laya_skip`):** avaliada **antes** da cotacao preliminar na Deriv; quando bloqueado, estima o piso localmente para telemetria, registra `SKIP | ...` e nao desperdicam chamadas `proposal`.
7. **Dimensionamento (`size_kelly_recovery`):** consulta payout preliminar, calcula Kelly fracionado com teto base (`max_base_pct`), piso de 1% e recuperacao por payout; saldo e limites sao revalidados na proposta final (`resize_laya_proposal`), sincronizando `orch.state.balance` e respeitando teto absoluto de 5%.
8. **Telemetria (`cycle_indicators_logger.py`):** imprime `INDICATORS | R_10 M5` (`TENDENCIA`, `MOMENTUM`, `DIRECAO`, `OSCILADORES`, `VOLATILIDADE`, `ESTRUTURA`), `LAYA_DECISION` e `MODELO` com 2 casas decimais.
9. **Execucao e Stop-Win:** `ExecutionManager.execute_laya_order` envia o contrato Rise/Fall de 5 minutos, reconcilia `SETTLE` com `balance_after` real e encerra graciosamente (`exit 0`) quando `realized_session_pnl >= stop_win_pct`.

## Integridade de Dados e Inferencia

- `StreamHandler.get_closed_candles(symbol, 300)` seleciona o buffer micro M5, ordena e deduplica epochs e exclui velas ainda abertas. O buffer diario nunca substitui o M5.
- Cada ciclo captura uma unica serie imutavel (ate 120 velas fechadas), reutilizada ate o registro da barra processada.
- `LAYA_CONTEXT` registra granularidade, quantidade, epoch, assinatura SHA-256, tokens consumidos e `dropped=0`. Respostas truncadas sao rejeitadas.
- A divida e os IDs liquidados integram o snapshot de risco; reconexao ou reconciliacao nao reaplicam resultados.
