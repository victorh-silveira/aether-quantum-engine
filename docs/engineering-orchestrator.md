# Orquestrador e ciclo

Ciclo operacional do motor. Inventario de arquivos: [`structure.md`](structure.md) §orchestrator. Arquitetura: [`arquitetura.md`](arquitetura.md) §3.

## Relogio e triplo OHLC

- Fronteira / ciclo: **`signature_boundary_seconds` / `cycle_interval_seconds` = 60 s** (prefixos legados `m5` / `m15` no codigo)
- MACRO OHLC: **300 s** (`data_handler.granularity`)
- MICRO OHLC (TCN decisor): **60 s** (`data_handler.micro_granularity`)
- Contrato Deriv RISE_FALL: **30 s** (`risk_management.params.duration`) — hibrido: label/treino em 1 barra micro 60 s
- MINI OHLC: **60 s** (`data_handler.mini_granularity`)
- MILI: tick flow (velocity/acceleration), nao barra OHLC
- Sync inicial: `stream_sync_start.py` (historico MACRO+MICRO+MINI + subscribe candles/ticks)
- Proporcao MACRO:MICRO **1:5** (300:60)
- Pos-settlement: `post_settlement_is_trading_wait_seconds` **35**; `settlement_tolerance_window_seconds` **60**

## Pipeline do ciclo

```text
warmup/buffer → training_gate → collect decisoes DL
  → resolve direcao + scale_vision → rank/pick → ExecutionManager
  → settlement worker → post_settlement → proxima fronteira
```

| Etapa | Onde |
|-------|------|
| Run loop | `orchestrator_run_loop.py`, `engine_session.py` |
| Assinatura dados | `orchestrator_data_signature.py` |
| Collect | `execution_collect*.py` |
| Direcao / escalas | `execution_direction_resolver.py`, `execution_scale_vision.py`, `execution_scale_micro.py`, `execution_scale_adapt.py`, `execution_scale_sizing.py` |
| Execucao | `execution_manager.py`, `execution_orders.py` |
| Settlement | `settlement_*.py`, `orchestrator_settlement_queue.py` |
| Pos-liquidacao | `post_settlement_*.py` |
| Persistencia | `orchestrator_persistence.py`, `orchestrator_atomic_state.py` |
| Watchdog | `watchdog_service.py` |

## Scale vision (MACRO/MICRO/MINI/MILI)

SSOT: `orchestrator.execution.scale_vision`. Escopo 1: **sem veto de sinal / sem SKIP por escala**. Sob `raw_extreme`/fita forte, adapta ao consenso da fita; sob **retracao** (`adapt_on_retraction`), adapta ao lado vivo `mi_curr`+MILI mesmo com par MINI rachado.

| Campo | Papel |
|-------|-------|
| MICRO | Direcao TCN do ciclo (telemetria `tcn_direction`) |
| MACRO | Slope dos closes (janela `slope_bars`) |
| MINI / MICRO bar | Vela **anterior** + **atual** open→close (`use_last_bar`) |
| MILI | Direcao do tick flow |
| `scale_micro_regime` | `explosion` / `retraction` / `chop` |
| `scale_tape_consensus` | Maioria da fita (`adapt_min_votes`) |
| `scale_adapted` | Par MINI + raw/forte **ou** retracao vs TCN; Kelly sync ao lado exec |
| Soft sizing | Discord/adapt/retracao/chop+mili_oppose → `kelly_mult_discord` + `scale_force_explore` |
| Soft cover | Pending material + `pending_waives_scale_explore` → soft cover/DAL sob `max_safe_stake_pct` (waive discord) |

Log: `SCALE || … tape=… micro=… adapted=0|1` e IND: `SCALE: tcn=… tape=… micro=… adapted=…`  
CLUSTER TF: `resolve_cluster_timeframe` prefere `micro_granularity` → tipicamente **M2**.

Nao confundir com `raw_extreme` (calibracao DL): limiares `tcn_macro_*_override` sao de **raw TCN**, nao da escala MACRO OHLC. Ver [`engineering-deep-learning.md`](engineering-deep-learning.md).

## Gates de fase

- **FASE TREINO:** sem ordens ate modelos da sessao prontos
- **FASE OPERACAO:** `mandatory_trade_each_cycle: true`, `force_trade_every_cycle: false`
- Lock/barreira serializa inferencia, liquidacao e persistencia

## Diagnostico rapido

| Sintoma | Ver |
|---------|-----|
| Ciclo nao dispara | signature, warmup buffer (MACRO/MICRO/MINI), idle watchdog |
| So EXEC_EMPTY | bloqueio tecnico / Kelly — processo pode estar correto |
| Stake baixo com SCALE discord/adapt | `kelly_mult_discord` + `max_stake_pct_discord` (esperado) |
| RECOVER/EXPLORE_DAL nao arma | `scale_force_explore` / `scale_adapted` forca Kelly |
| Lado ≠ TCN no EXEC | `scale_adapted` + `tape_vs_tcn` sob `raw_extreme` |
| Travado apos trade | settlement queue / post_settlement |
| Reconnect loop | watchdog stale + cooldown |

Skill: `aether-cycle-debug`.
