# Playbook trader senior — binarias M2 (`R_10`; OHLC 120s)

Postura operacional (**escopo 1.1** + arquitetura continua R_10): TCN resolve CALL/PUT; meta/edge/indicadores sao telemetria; SCALE adapta lado sem SKIP por escala. SKIP = tecnico (inclui **neg_edge** hard sob mandato **2026-08-09**); loss-clf alto = **FLIP** CALL↔PUT (`p_loss>=0.90`, `veto_ready`); catálogo soft `signal_skip` mini/cal/chop e loss-clf faixa media = soft Kelly (sem revenge sizing pos-LOSS).

Universo: **Volatility 10** (`R_10`) — **M2** (contrato **2 m**, ciclo **60 s**, micro/MINI **120 s**, macro **3600 s**).

Hierarquia: TCN Cal/Margin → CALL/PUT (SCALE adapt) → soft `signal_skip` / loss-clf soft → loss-clf **FLIP** / chop soft → **neg_edge hard** → Kelly/caps. **Proibido** reabrir quality gate amplo (RSI/price_zone/SIDE_EQ block).

## Quando operar

| Lado | Condicoes tipicas |
|------|-------------------|
| CALL | TCN CALL, ou fita adapta para CALL |
| PUT | TCN PUT, ou fita adapta para PUT |
| SKIP tecnico | `training` / `data` / `deploy` / `predict_error` / `neg_edge`, warm-up, stop-win, broker |
| Soft sinal 1.1 | `mini_pair_oppose` / `cal_margin` / loss-clf faixa media → soft Kelly; lado pos-LOSS permanece no TCN |
| Flip loss-clf | `p_loss >= hard_p_loss_floor` (**0.90**) e `veto_ready`; SCALE confirma TCN bloqueia (seed nao anula SCALE por Cal); Cal discord exige `|cal-0.5|>=0.03`; pos-FLIP edge < `min_edge_execute` → `FLIP_BLOCK:neg_edge` + soft; seed so com SCALE discord (ou Cal forte sem tape TCN) |
| Chop soft | ADX &lt; **0.22** e (Hurst ∈ [**0.47**, **0.53**] ou SCALE micro=chop) → soft Kelly **0.55**; log `REGIME \|\| CHOP_SOFT` |
| `majority_votes` | Mais votos PUT ou CALL (tape/mili/RSI vs TCN) → adapta lado |

## Catalogo SKIP tecnico

| Razao | Significado |
|-------|-------------|
| `training` | Treino de sessao em andamento |
| `data` | Buffer/historico insuficiente |
| `deploy` | Checkpoint sem `deploy_ok` |
| `predict_error` | Falha de inferencia |
| `neg_edge` | Edge Cal do lado &lt; `min_edge_execute` (**mandato 2026-08-09**) |
| Kelly `EXEC_PAUSE` | `stop_win` / `bankroll_below_stake_min` (sizing; **sem** `kelly_no_edge`) |

## Catalogo sinal / ML (soft + neg_edge hard)

| Razao | Significado |
|-------|-------------|
| `mini_pair_oppose` | Par MINI unanime ≠ lado executado → **sempre** soft Kelly (`mini_pair_soft_kelly_mult` **0.55**); sem hard SKIP |
| `cal_margin` | `direction_margin` &lt; `min_direction_margin` → soft Kelly; waive com pending material |
| `loss_clf_soft` | Container loss-clf: atenua Kelly; log `LOSS_CLF \|\| SOFT` |
| `loss_clf_flip` | `p_loss >= 0.90` + `veto_ready` + waivers seed/scale + edge pos-FLIP ≥ floor → FLIP (`from→to`/`why` em `[GATES]`); senao `FLIP_BLOCK` (seed/scale/neg_edge) + soft |
| `regime_chop` | ADX/Hurst (ou SCALE chop) → soft Kelly (`chop_soft_kelly_mult` **0.55**); log `REGIME \|\| CHOP_SOFT` |
| `neg_edge` | Edge Cal do lado &lt; `min_edge_execute` → **hard-skip** (`gate_reason=neg_edge`, EXEC_EMPTY); log `EDGE \|\| NEG_HARD` (**mandato 2026-08-09**) |

Quality gate amplo (RSI/discordance/price zone/SIDE_EQ block) permanece **fora** do codigo. Chop permanece soft Kelly; **neg_edge** e hard-skip sob mandato **2026-08-09**; flip loss-clf permanece sob mandato **2026-08-07**.

## Escalas MACRO / MICRO / MINI / MILI

Triplo OHLC + ticks: telemetria, **adaptacao de lado** e soft Kelly — **nunca SKIP por escala**.

| Escala | Fonte | Papel |
|--------|-------|-------|
| MACRO | OHLC **3600 s** | Slope closes — contexto |
| MICRO | OHLC **120 s** | TCN + last-bar na fita (M2) |
| MINI | OHLC **120 s** | Last-bar anterior+atual + slope (M2) |
| MILI | Tick flow | Fluxo intrabar |

Log: `SCALE || … mi_prev=… mi_cur=… tape=… micro=retract|explos|chop adapted=0|1` e no IND `SCALE: tcn=… tape=… votes=C#/P# …`.  
Adapt: **majority_votes** (TCN/tape/mili/RSI) sem hold Cal; tape sob `raw_extreme`; regimes **retracao** / **explosao** / **mili+tape** (mili+tape **nao** adapta em micro=chop; `adapt_mili_tape_skip_chop`). `adapt_allow_strong_tape` **false**. Kelly `kelly_p_floor` **0.55**; explore piso `neutral_bankroll_pct` **0.25%** (M2); `fraction` **0.08**; RECOVER cover **2x** (`cover_multiple`); teto **5%**; payout **0.72**; stop-win Kelly **4 ciclos/1h**. Sem `kelly_no_edge` / sem SKIP por escala / **sem** zona cinza.
Contrato Deriv **2 m**; label TCN = 1 barra micro (**120 s**).

## `raw_extreme` (anti-override)

Modo de calibracao `raw_extreme` **substituiu** `tcn_macro_override`: raw extremo **nao** substitui Cal; Kelly usa probabilidade calibrada. Chaves SSOT `tcn_macro_call_override` / `tcn_macro_put_override` so limiam raw extremo — **nao** sao timeframe MACRO.

## Edge Cal (identidade Kelly)

Edge = `p_side * (1 + b) - 1` com **p = Cal** do lado (nunca raw). Com `b=0.72`: breakeven `be ≈ 0.581`; floor `min_edge_execute` **0.04** exige Cal ≳ **0.605**. Cal~0.53 → Edge~−0.08 e hard-skip `neg_edge` e matematica correta. `[CLUSTER]` inclui `raw_edge` + `be` sempre que Edge e exibido (CLUSTER sai antes do hard-skip); `[GATES]` sob `neg_edge`/EXEC_EMPTY repete o gap. So telemetria — raw extremo ≠ Kelly.

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
- `risk_management.kelly.neutral_bankroll_pct` / `min_stake_pct` **0.0025** — explore piso **0.25%** banca (M2)
- `risk_management.kelly.fraction` **0.08** — Kelly fracionario baixo (alta frequencia)
- `risk_management.soft_recovery.infeasible_force_explore: true` — `RECOVERY_INFEASIBLE` (ou cover ≥ cap) forca EXPLORE Kelly, nao DAL no teto
- `soft_recovery.live_evidence_force_explore_*` — linear≥3 com `live_wr` fraco forca EXPLORE (evita DAL L3+ enquanto ACC de checkpoint ainda passa)
- `pending_waives_scale_explore: true` — com pending material, soft cover/DAL nao e short-circuitado por discord/adapt
- `risk_management.soft_recovery.amort_cycles_*` **1/1** — stake RECOVER = cover `pending/payout` (sem progressao geometrica sobre o cover); caps `max_safe_stake_*`
- `orchestrator.execution.side_equilibrium.enabled: true` (soft sizing only)
- `orchestrator.execution.scale_vision` (adaptacao de fita + soft sizing; sem SKIP por escala)
- `orchestrator.execution.signal_skip` (1.1 soft Kelly; sem flip pos-LOSS)

Ver doutrina [`llm-trading-doctrine.md`](llm-trading-doctrine.md), [`deriv-indices-algorithm.md`](deriv-indices-algorithm.md), [`infra-docker.md`](infra-docker.md) e [`engineering-settings-ssot.md`](engineering-settings-ssot.md).
