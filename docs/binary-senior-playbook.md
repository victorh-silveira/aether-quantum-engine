# Playbook trader senior — binarias M2 (`R_10`; OHLC 120s)

Postura operacional (**escopo 1.1** + arquitetura continua R_10): TCN ancora Cal; **fusao EV** multi-escala (`fusion_*`) escolhe CALL/PUT; meta/edge/indicadores sao telemetria; SCALE telemetria + adapt quando fusao nao substitui. SKIP = tecnico (treino/dados/deploy/broker; **sem** EMPTY de sinal por `neg_edge` no default); loss-clf alto alimenta fusao/FLIP; catálogo soft `signal_skip` mini/cal/chop/`neg_edge` = soft Kelly (sem revenge sizing pos-LOSS).

Universo: **Volatility 10** (`R_10`) — **M2** (contrato **2 m**, ciclo **120 s**, micro/MINI **120 s**, macro **3600 s**).

Hierarquia: TCN Cal/Margin → SCALE dirs → soft `signal_skip` / loss-clf → **fusao EV** (argmax CALL/PUT) → chop soft → **neg_edge** soft Kelly → Kelly/caps. **Proibido** reabrir quality gate amplo (RSI/price_zone/SIDE_EQ block).

## Quando operar

| Lado | Condicoes tipicas |
|------|-------------------|
| CALL | TCN CALL, fusao EV_CALL >= EV_PUT, ou fita/adapt |
| PUT | TCN PUT, fusao EV_PUT > EV_CALL, ou fita/adapt |
| SKIP tecnico | `training` / `data` / `deploy` / `predict_error`, warm-up, stop-win, broker; `neg_edge` hard so com override ou seed+edge &lt; **−0.12** |
| Soft sinal 1.1 | `mini_pair_oppose` / `cal_margin` / loss-clf; chop soft off no experimento; `neg_edge` soft (`deep_floor` **−0.12**); fusao usa `fusion_p_eff`; EV fraco → soft Kelly **0.40** (seed+ambos EV&lt;0 → **0.25**); `invert_exec_side` **false** |
| Fusao multi-escala | `fusion_enabled`: p_eff (Cal + MACRO/vela/MINI/MILI/tape + loss continuo + meta **0.10**); `fusion_loss_weight` **0.45** so com `auto=1`; `fusion_block_when_tcn_pos_edge` **true** preserva TCN so se Cal **e** raw +EV; `fusion_block_when_tcn_candle_agree` **true** preserva TCN se vela==TCN (`why=tcn_candle_agree`); telemetria `[GATES] \|\| FUSION` + `fusion_ev_*` / `fusion_p_eff` |
| Flip loss-clf | `p_loss >= hard_p_loss_floor` (**0.90**) e `veto_ready` + `flip_require_auto_learn` (**true**: seed so SOFT); **bloqueia FLIP** se Edge Cal **e** raw_edge do TCN >= floor (`FLIP_BLOCK:tcn_edge`; Cal+/raw− nao trava); sob seed, vela fechada == TCN bloqueia (`FLIP_BLOCK:seed_candle`; `p_ovr` nao fura); seed edge min **−0.08**; live `flip_waive_edge_min` **−1.0**; vela no alvo floor **0.85** so se TCN fraco |
| Chop soft | ADX &lt; **0.22** e (Hurst ∈ [**0.47**, **0.53**] ou SCALE micro=chop) → soft Kelly **0.55**; log `REGIME \|\| CHOP_SOFT` |
| `majority_votes` | Mais votos PUT ou CALL (tape/mili/RSI vs TCN) → adapta lado (se `fusion_replace_adapt_flip` **false**) |

## Catalogo SKIP tecnico

| Razao | Significado |
|-------|-------------|
| `training` | Treino de sessao em andamento |
| `data` | Buffer/historico insuficiente |
| `deploy` | Checkpoint sem `deploy_ok` |
| `predict_error` | Falha de inferencia |
| `neg_edge` | Hard se `neg_edge_hard_skip` **true** ou seed+edge &lt; `neg_edge_deep_edge_floor` (**−0.12**); senao soft Kelly |
| Kelly `EXEC_PAUSE` | `stop_win` / `bankroll_below_stake_min` (sizing; **sem** `kelly_no_edge`) |

## Catalogo sinal / ML (soft + neg_edge soft)

| Razao | Significado |
|-------|-------------|
| `mini_pair_oppose` | Par MINI unanime ≠ lado executado → **sempre** soft Kelly (`mini_pair_soft_kelly_mult` **0.55**); sem hard SKIP |
| `cal_margin` | `direction_margin` &lt; `min_direction_margin` → soft Kelly; waive com pending material |
| `loss_clf_soft` | Container loss-clf: atenua Kelly; log `LOSS_CLF \|\| SOFT` |
| `loss_clf_flip` | `p_loss >= 0.90` + `veto_ready` + waivers; seed: `seed_candle` / `flip_seed_waive_edge_min` **−0.08**; live: `flip_waive_edge_min` **−1.0** → FLIP (`from→to`/`why` em `[GATES]`); senao `FLIP_BLOCK` (seed_candle/seed/scale/neg_edge/tcn_edge) + soft |
| `regime_chop` | ADX/Hurst (ou SCALE chop) → soft Kelly (`chop_soft_kelly_mult` **0.55**); log `REGIME \|\| CHOP_SOFT` |
| `neg_edge` | Edge Cal do lado &lt; `min_edge_execute` → soft Kelly (`neg_edge_soft_min_edge` **−1.0**; seed mult **0.25**); hard so seed+edge &lt; **−0.12** ou override |

Quality gate amplo (RSI/discordance/price zone/SIDE_EQ block) permanece **fora** do codigo. Loss-clf live apos ~8–16 settles mistos (`auto=1`); restart container apos mudar `LOSS_BOOTSTRAP_EXIT_N`.
## Escalas MACRO / MICRO / MINI / MILI

Triplo OHLC + ticks: telemetria, **adaptacao de lado** e soft Kelly — **nunca SKIP por escala**.

| Escala | Fonte | Papel |
|--------|-------|-------|
| MACRO | OHLC **3600 s** | Slope closes — contexto |
| MICRO | OHLC **120 s** | TCN + last-bar na fita (M2) |
| MINI | OHLC **120 s** | Last-bar anterior+atual + slope (M2) |
| MILI | Tick flow | Fluxo intrabar |

Log: `SCALE || … mi_prev=… mi_cur=… tape=… micro=retract|explos|chop adapted=0|1` e no IND `SCALE: tcn=… tape=… votes=C#/P# …`.  
Adapt: **majority_votes** (TCN/tape/mili/RSI) sem hold Cal; tape sob `raw_extreme`; regimes **retracao** / **explosao** / **mili+tape** (mili+tape **nao** adapta em micro=chop; `adapt_mili_tape_skip_chop`). `adapt_allow_strong_tape` **false**. Kelly `kelly_p_floor` **0.55**; com fusao ancora em `fusion_p_eff`; explore piso `neutral_bankroll_pct` **0.25%** + `explore_stake_scale_floor` **0.40**; `fraction` **0.08**; RECOVER cover amortizado (`cover_multiple` **1.50**, amort **2–4**; `f*` so gate); damping stop-win inicio **1.0** / perto-meta **0.50**; teto linear3 **2.5%**; payout **0.72**; stop-win Kelly **4 ciclos/1h**. Sem `kelly_no_edge` / sem SKIP por escala / **sem** zona cinza.
Contrato Deriv **2 m**; label TCN = 1 barra micro (**120 s**).

## `raw_extreme` (anti-override)

Modo de calibracao `raw_extreme` **substituiu** `tcn_macro_override`: raw extremo **nao** substitui Cal; Kelly usa probabilidade calibrada. Chaves SSOT `tcn_macro_call_override` / `tcn_macro_put_override` so limiam raw extremo — **nao** sao timeframe MACRO.

## Edge Cal (identidade Kelly)

Edge = `p_side * (1 + b) - 1` com **p = Cal** do lado (nunca raw). Com `b=0.72`: breakeven `be ≈ 0.581`; floor `min_edge_execute` **0.04** exige Cal ≳ **0.605**. Cal~0.53 → Edge~−0.08 e soft Kelly `neg_edge` (nao EMPTY no default). `[CLUSTER]` inclui `raw_edge` + `be` sempre que Edge e exibido; `[GATES]` sob soft/EXEC repete o gap. So telemetria — raw extremo ≠ Kelly.

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
- Caps Kelly / `max_safe_stake_*` (`max_safe_stake_pct` **0.05**; linear2 **0.04**; linear3 **0.025**)
- `risk_management.soft_recovery.cover_multiple: 1.50` — RECOVER = cover amortizado `pending/payout/amort`
- `risk_management.kelly.neutral_bankroll_pct` / `min_stake_pct` **0.0025** — explore piso **0.25%** banca (M2)
- `risk_management.kelly.fraction` **0.08** — Kelly fracionario baixo (alta frequencia)
- `risk_management.soft_recovery.infeasible_force_explore: true` — `RECOVERY_INFEASIBLE` (ou cover ≥ cap) forca EXPLORE Kelly, nao DAL no teto
- `soft_recovery.live_evidence_force_explore_*` — linear≥3 com `live_wr` &lt; **0.62** forca EXPLORE (evita DAL L3+ enquanto ACC de checkpoint ainda passa)
- `pending_waives_scale_explore: true` — com pending material, soft cover/DAL nao e short-circuitado por discord/adapt
- `risk_management.soft_recovery.amort_cycles_*` **2/4** — stake RECOVER = cover amortizado (sem progressao geometrica); caps `max_safe_stake_*`
- `orchestrator.execution.side_equilibrium.enabled: true` (soft sizing only)
- `orchestrator.execution.scale_vision` (adaptacao de fita + soft sizing; sem SKIP por escala)
- `orchestrator.execution.signal_skip` (1.1 soft Kelly; sem flip pos-LOSS)

Ver doutrina [`llm-trading-doctrine.md`](llm-trading-doctrine.md), [`deriv-indices-algorithm.md`](deriv-indices-algorithm.md), [`infra-docker.md`](infra-docker.md) e [`engineering-settings-ssot.md`](engineering-settings-ssot.md).
