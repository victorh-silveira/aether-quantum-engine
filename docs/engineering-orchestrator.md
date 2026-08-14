# Orquestrador e ciclo

Ciclo operacional do motor. Inventario de arquivos: [`structure.md`](structure.md) §orchestrator. Arquitetura: [`arquitetura.md`](arquitetura.md) §3.

## Relogio e triplo OHLC

- Fronteira / ciclo: **`signature_boundary_seconds` / `cycle_interval_seconds` = 180 s** (sync com fecho da vela M3; contrato Deriv **N × 3 min**); `exec_empty_retry_seconds` **180**
- Cache DL (`dl_predict_cache`): path **eager** (Triton off) **sempre** re-infere; path Triton chaveia `cycle_id` + `boundary_epoch` (nao reusa entry de outro ciclo)
- Tick live: antes do TCN, `patch_forming_bar_with_live_tick` injeta o ultimo preco do `TickBuffer` no close/high/low da vela M3 em formacao; `patch_forming_bar_microstructure` sobrescreve a ultima linha de micro live; snapshot `_patched_ohlc` alimenta SCALE/flow no mesmo ciclo
- `DL: inferencia em cuda` e `log_device_once` no load do modelo — **nao** um log por ciclo
- LOSS_CLF: predict HTTP a cada `_finalize`; log dedupe por `loss_clf_*:{cycle_id}`; `feature_dim` **24**; hard FLIP floor SSOT **0.90** + `flip_require_auto_learn` **true**
- MACRO OHLC: **7200 s** (`data_handler.granularity`)
- MICRO OHLC (TCN decisor): **180 s** (`data_handler.micro_granularity`) — M3
- Contrato Deriv RISE_FALL: **N × 3 min** (`risk_management.params.duration`) — N eleito no launch-train (`horizon_sweep`, grade 1/2/3/5); label = **N barras** micro; frequencia maxima ≈ 1 trade / contrato (ciclo bloqueado com contrato aberto)
- MINI OHLC: **180 s** (`data_handler.mini_granularity`) — alinhado ao M3
- MILI: tick flow (velocity/acceleration), nao barra OHLC
- Sync inicial: `stream_sync_start.py` (historico MACRO+MICRO+MINI + subscribe candles/ticks)
- Proporcao MACRO:MICRO **40:1** (7200:180)
- Pos-settlement: `post_settlement_is_trading_wait_seconds` **90**; `settlement_tolerance_window_seconds` **90**; `post_settlement_cycle_timeout_seconds` **1200**; `watchdog_stale_tick_seconds` **300**

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
| Direcao / escalas | `execution_direction_resolver.py`, `execution_direction_fusion.py`, `execution_scale_vision.py`, `execution_scale_micro.py`, `execution_scale_adapt.py`, `execution_scale_sizing.py`, `execution_signal_skip.py` |
| Execucao | `execution_manager.py`, `execution_orders.py` |
| Settlement | `settlement_*.py`, `orchestrator_settlement_queue.py` |
| Pos-liquidacao | `post_settlement_*.py` |
| Persistencia | `orchestrator_persistence.py`, `orchestrator_atomic_state.py` |
| Watchdog | `watchdog_service.py` |

## Scale vision (MACRO/MICRO/MINI/MILI)

SSOT: `orchestrator.execution.scale_vision` + `signal_skip` (escopo **1.1**). SCALE adapta lado sem SKIP por escala; apos adapt, catálogo minimo atenua com soft Kelly (`mini_pair_oppose` / `cal_margin`) — **sem** hard SKIP de sinal, **sem** flip pos-LOSS e **sem** zona cinza/`hold_cal_*`. Ordem adapt: **majority_votes** (TCN/tape/mili/mini_pair/RSI) → tape/`raw_extreme` → regimes explosao/retracao/mili+tape. `adapt_allow_strong_tape` **false**.

| Campo | Papel |
|-------|-------|
| MICRO | Direcao TCN do ciclo (telemetria `tcn_direction`) |
| MACRO | Slope dos closes (janela `slope_bars`) |
| MINI / MICRO bar | Vela **anterior** + **atual** open→close (`use_last_bar`) |
| MILI | Direcao do tick flow |
| `scale_micro_regime` | `explosion` / `retraction` / `chop` |
| `scale_tape_consensus` | Maioria da fita (`adapt_min_votes`) |
| `scale_adapted` | Flip so sob `raw_extreme` ou margem fraca + regimes; Kelly sync ao lado exec |
| Soft sizing | Discord/adapt/retracao/chop+mili_oppose → `kelly_mult_discord` + `scale_force_explore` |
| Soft cover | Pending material + `pending_waives_scale_explore` → cover fino; `adapted_force_explore` bloqueia DAL L2+ |

Log: `SCALE || … tape=… micro=… adapted=0|1` e IND: `SCALE: tcn=… tape=… micro=… adapted=…`  
CLUSTER TF: `resolve_cluster_timeframe` prefere `micro_granularity` → tipicamente **M3**.

Nao confundir com `raw_extreme` (calibracao DL): limiares `tcn_macro_*_override` sao de **raw TCN**, nao da escala MACRO OHLC. Ver [`engineering-deep-learning.md`](engineering-deep-learning.md).

## Gates de fase

- **FASE TREINO:** sem ordens ate modelos da sessao prontos
- **FASE OPERACAO:** `mandatory_trade_each_cycle: false`, `force_trade_every_cycle: false`, `invert_exec_side: false`, `online_training: false`
- Lock/barreira serializa inferencia, liquidacao e persistencia

## Diagnostico rapido

| Sintoma | Ver |
|---------|-----|
| Ciclo nao dispara | signature, warmup buffer (MACRO/MICRO/MINI), idle watchdog |
| So EXEC_EMPTY | bloqueio tecnico / Kelly — processo pode estar correto |
| Stake baixo com SCALE discord/adapt | `kelly_mult_discord` + `max_stake_pct_discord` (esperado) |
| RECOVER/EXPLORE_DAL nao arma | `adapted_force_explore` / ACC / live_wr / `scale_force_explore` |
| Lado ≠ TCN no EXEC | `scale_adapted` via **majority_votes**, tape/`raw_extreme` ou regimes (sem hold Cal) |
| Travado apos trade | settlement queue / post_settlement |
| Reconnect loop | watchdog stale + cooldown |

Skill: `aether-cycle-debug`.
