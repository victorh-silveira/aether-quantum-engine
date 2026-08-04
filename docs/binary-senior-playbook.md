# Playbook trader senior — binarias 120s (`R_10`)

Postura operacional (mandato escopo 1): **pipeline sem vetos de sinal/qualidade**. TCN resolve CALL/PUT; meta/edge/indicadores sao telemetria. SKIP apenas por bloqueio tecnico.

Hierarquia: TCN Cal/Margin (telemetria) → CALL/PUT (pode adaptar a fita sob `raw_extreme`) → Kelly/caps (soft: SIDE_EQ + scale_vision). Escopo 1: **sem veto de sinal / sem SKIP por escala**.

## Quando operar

| Lado | Condicoes tipicas |
|------|-------------------|
| CALL | TCN CALL, ou fita adapta para CALL sob `raw_extreme` |
| PUT | TCN PUT, ou fita adapta para PUT sob `raw_extreme` |
| SKIP tecnico | `training` / `data` / `deploy` / `predict_error`, warm-up, stop-win, broker |

## Catalogo SKIP tecnico

| Razao | Significado |
|-------|-------------|
| `training` | Treino de sessao em andamento |
| `data` | Buffer/historico insuficiente |
| `deploy` | Checkpoint sem `deploy_ok` |
| `predict_error` | Falha de inferencia |
| Kelly `EXEC_PAUSE` | `kelly_no_edge` / stake 0 (sizing, nao veto de direcao) |

Vetos de sinal removidos do codigo: Hurst/ADX/RSI/discordance/adverse path/price zone, quality gate (cal floor, margin, meta edge, starvation), SIDE_EQ bloqueante, senior skip catalog.

## Escalas MACRO / MICRO / MINI / MILI

Triplo OHLC + ticks: telemetria, **adaptacao de lado** (fita vs TCN sob `raw_extreme`) e soft Kelly — **nunca SKIP por escala**.

| Escala | Fonte | Papel |
|--------|-------|-------|
| MACRO | OHLC **600 s** | Slope closes — contexto |
| MICRO | OHLC **120 s** | TCN + last-bar na fita |
| MINI | OHLC **60 s** | Last-bar anterior+atual + slope |
| MILI | Tick flow | Fluxo intrabar |

Log: `SCALE || … mi_prev=… mi_cur=… tape=… adapted=0|1` e no IND `SCALE: tcn=… tape=… mi_p=… mi=…`.  
Modulos: `execution_scale_vision.py`, `execution_scale_adapt.py`, `execution_scale_sizing.py`, sync em `stream_sync_start.py`.  
Adapt: consenso da fita usa **vela anterior e atual** (MINI/MICRO) + MILI; `adapt_min_votes` (**2**) + `adapt_require_raw_extreme`.  
Soft: discord/adapt → `kelly_mult_discord` (**0.35**) + `scale_force_explore` (corta DAL) + `max_stake_pct_discord` (**0.005**). CLUSTER TF prefere micro (`M2`).

## `raw_extreme` (anti-override)

Modo de calibracao `raw_extreme` **substituiu** `tcn_macro_override`: raw extremo **nao** substitui Cal; Kelly usa probabilidade calibrada. Chaves SSOT `tcn_macro_call_override` / `tcn_macro_put_override` so limiam raw extremo — **nao** sao timeframe MACRO.

## Vies CALL/PUT (pos-escopo 1)

Viés estrutural de lado (ex.: PUT collapse, `label_call_frac` longe de 0.5) **nao** se corrige reintroduzindo veto de sinal nem quality gate. Correção no treino + sizing:

| Camada | O que faz |
|--------|-----------|
| Treino | `deep_learning.sample_weighting` (class balance + recency half-life) |
| Deploy | `reject_majority_collapse` / `max_label_call_frac_bias` / `min_minority_recall` |
| Live | SIDE_EQ **soft Kelly** (`execution_side_eq_sizing`): atenua stake no lado toxico; **nunca** SKIP/veto de direcao; `side_eq_blocked` permanece false |

Lado enviesado no log live ≠ SKIP tecnico. SKIP continua so `training`/`data`/`deploy`/`predict_error` (+ Kelly pause/caps).

## Knobs SSOT restantes (senior)

- `force_trade_every_cycle: false` (proibido como “fix”)
- `min_validation_accuracy_gate: 0.53` (treino/deploy)
- Caps Kelly / `max_safe_stake_*`
- `orchestrator.execution.side_equilibrium.enabled: true` (soft sizing only)
- `orchestrator.execution.scale_vision` (adaptacao de fita + soft sizing; sem veto/SKIP)

Ver doutrina [`llm-trading-doctrine.md`](llm-trading-doctrine.md) e [`engineering-settings-ssot.md`](engineering-settings-ssot.md).
