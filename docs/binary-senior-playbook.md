# Playbook trader senior — binarias M15 (`OTC_SPC`; OHLC 900s)

Postura operacional (**escopo 1.1**, mandato): TCN resolve CALL/PUT; meta/edge/indicadores sao telemetria; SCALE adapta lado sem SKIP por escala. SKIP = tecnico; catálogo `signal_skip` / loss-clf = **somente soft Kelly** (sem EXEC_EMPTY de sinal; sem flip pos-LOSS).

Universo: **S&P 500 OTC** (`OTC_SPC`) — **somente M15** (contrato **15 m**, ciclo/micro/MINI **900 s**, macro **3600 s**, proporcao 1:5).

Hierarquia: TCN Cal/Margin → CALL/PUT (SCALE adapt) → soft `signal_skip` / loss-clf → Kelly/caps (SIDE_EQ + scale soft). **Proibido** restaurar quality gate / Hurst/ADX/RSI/price_zone.

## Quando operar

| Lado | Condicoes tipicas |
|------|-------------------|
| CALL | TCN CALL, ou fita adapta para CALL |
| PUT | TCN PUT, ou fita adapta para PUT |
| SKIP tecnico | `training` / `data` / `deploy` / `predict_error`, warm-up, stop-win, broker |
| Soft sinal 1.1 | `mini_pair_oppose` / `cal_margin` / loss-clf → soft Kelly; lado pos-LOSS permanece no TCN |
| `majority_votes` | Mais votos PUT ou CALL (tape/mili/RSI vs TCN) → adapta lado; IND mostra `votes=C#/P#` |

## Catalogo SKIP tecnico

| Razao | Significado |
|-------|-------------|
| `training` | Treino de sessao em andamento |
| `data` | Buffer/historico insuficiente |
| `deploy` | Checkpoint sem `deploy_ok` |
| `predict_error` | Falha de inferencia |
| Kelly `EXEC_PAUSE` | `stop_win` / `bankroll_below_stake_min` (sizing; **sem** `kelly_no_edge`) |

## Catalogo sinal / ML (soft; sem EXEC_EMPTY de sinal)

| Razao | Significado |
|-------|-------------|
| `mini_pair_oppose` | Par MINI unanime ≠ lado executado → **sempre** soft Kelly (`mini_pair_soft_kelly_mult` **0.55**); sem hard SKIP |
| `cal_margin` | `direction_margin` &lt; `min_direction_margin` → soft Kelly; waive com pending material |
| `majority_votes` | Mais votos PUT ou CALL (tape/mili/RSI vs TCN) → adapta lado; IND mostra `votes=C#/P#` |
| `loss_clf_soft` | Container loss-clf: atenua Kelly; log `LOSS_CLF \|\| SOFT` |

Quality gate amplo (Hurst/ADX/RSI/discordance/price zone/SIDE_EQ block) permanece **fora** do codigo.

## Escalas MACRO / MICRO / MINI / MILI

Triplo OHLC + ticks: telemetria, **adaptacao de lado** e soft Kelly — **nunca SKIP por escala**.

| Escala | Fonte | Papel |
|--------|-------|-------|
| MACRO | OHLC **3600 s** | Slope closes — contexto (1:5 vs M15) |
| MICRO | OHLC **900 s** | TCN + last-bar na fita (M15) |
| MINI | OHLC **900 s** | Last-bar anterior+atual + slope (M15) |
| MILI | Tick flow | Fluxo intrabar |

Log: `SCALE || … mi_prev=… mi_cur=… tape=… micro=retract|explos|chop adapted=0|1` e no IND `SCALE: tcn=… tape=… votes=C#/P# …`.  
Adapt: **majority_votes** (TCN/tape/mili/RSI) sem hold Cal; tape sob `raw_extreme`; regimes **retracao** / **explosao** / **mili+tape**. `adapt_allow_strong_tape` **false**. Kelly `kelly_p_floor` **0.55**; explore piso `neutral_bankroll_pct` **2%**; RECOVER cover **2x** (`cover_multiple`); teto **5%**; payout **0.72**; stop-win Kelly **4 ciclos/1h**. Sem `kelly_no_edge` / sem SKIP por escala / **sem** zona cinza.
Contrato Deriv **15 m**; label TCN = 1 barra micro (**900 s**).

## `raw_extreme` (anti-override)

Modo de calibracao `raw_extreme` **substituiu** `tcn_macro_override`: raw extremo **nao** substitui Cal; Kelly usa probabilidade calibrada. Chaves SSOT `tcn_macro_call_override` / `tcn_macro_put_override` so limiam raw extremo — **nao** sao timeframe MACRO.

## Vies CALL/PUT (pos-escopo 1)

Viés estrutural de lado **nao** se corrige reintroduzindo veto de sinal nem quality gate. Correção no treino + sizing:

| Camada | O que faz |
|--------|-----------|
| Treino | `deep_learning.sample_weighting` (class balance + recency half-life) |
| Deploy | `reject_majority_collapse` / `max_label_call_frac_bias` / `min_minority_recall` |
| Live | SIDE_EQ **soft Kelly** (`execution_side_eq_sizing`): atenua stake no lado toxico; **nunca** SKIP/veto de direcao |

## Knobs SSOT restantes (senior)

- `force_trade_every_cycle: false` (proibido como “fix”)
- `min_validation_accuracy_gate: 0.53` (treino/deploy)
- Caps Kelly / `max_safe_stake_*` (`max_safe_stake_pct` **0.05**; linear2/3 **0.05**)
- `risk_management.soft_recovery.cover_multiple: 2.0` — RECOVER = 2× `pending/payout` (loss + win)
- `risk_management.kelly.neutral_bankroll_pct` / `min_stake_pct` **0.02** — explore obrigatorio **2%** banca
- `risk_management.soft_recovery.infeasible_force_explore: true` — `RECOVERY_INFEASIBLE` (ou cover ≥ cap) forca EXPLORE Kelly, nao DAL no teto
- `soft_recovery.live_evidence_force_explore_*` — linear≥3 com `live_wr` fraco forca EXPLORE (evita DAL L3+ enquanto ACC de checkpoint ainda passa)
- `pending_waives_scale_explore: true` — com pending material, soft cover/DAL nao e short-circuitado por discord/adapt
- `risk_management.soft_recovery.amort_cycles_*` **1/1** — stake RECOVER = cover `pending/payout` (sem progressao geometrica sobre o cover); caps `max_safe_stake_*`
- `orchestrator.execution.side_equilibrium.enabled: true` (soft sizing only)
- `orchestrator.execution.scale_vision` (adaptacao de fita + soft sizing; sem SKIP por escala)
- `orchestrator.execution.signal_skip` (1.1 soft Kelly; sem flip pos-LOSS)

Ver doutrina [`llm-trading-doctrine.md`](llm-trading-doctrine.md), [`deriv-indices-algorithm.md`](deriv-indices-algorithm.md), [`infra-docker.md`](infra-docker.md) e [`engineering-settings-ssot.md`](engineering-settings-ssot.md).
