# Playbook trader senior — binarias M5 (`1HZ75V`; OHLC 300s)

Postura operacional (**escopo 1.1** + arquitetura continua 1HZ75V): TCN ancora Cal; **fusao EV** multi-escala (`fusion_*`) escolhe CALL/PUT; depois **loss-clf FLIP** (ref TCN); meta/edge/indicadores sao telemetria; SCALE telemetria + adapt quando fusao nao substitui. SKIP tecnico = treino/dados/deploy/broker/stop-win; **neg_edge** com `neg_edge_hard_skip` **true** (SSOT): Edge `<= 0` → HARD SKIP; soft SOFT_SIZE so se `0 < Edge < min_edge_*` (**sem** Single-Strike); Z-score panic sempre hard; catálogo soft `signal_skip` mini/cal/chop = soft Kelly (sem revenge sizing pos-LOSS).

Universo: **Volatility 75 (1s) Index** (`1HZ75V`) — **M5** (contrato ops **5 m / M5**; label TCN **N=1** vela M5; ciclo **120 s**, micro/MINI **300 s**, macro **86400 s** (D1), ratio **1:288**).

Hierarquia: TCN Cal/Margin → SCALE dirs → soft `signal_skip` → **fusao EV** (argmax CALL/PUT) → **loss-clf FLIP** (ref TCN, ultimo) → **anti-loss com microestrutura estrita** (EMA Slope M5 + Zero Bypass Vela M5 + RSI Momentum) → chop soft → **neg_edge** (`gate_verdict` HARD/SOFT/ALLOW; Edge≤0 hard; subfloor soft sem Single-Strike; Trava Z-Score Pânico; `fusion_p_eff` so Kelly apos o gate) → Kelly Single-Strike / caps. Caveat: `fusion_loss_weight` **nao** ve o `p_loss` do mesmo ciclo (FLIP ocorre apos a fusao); sob seed, `loss_bonus` ja e **0**. **Proibido** reabrir quality gate amplo legado.

## Quando operar

| Lado | Condicoes tipicas |
|------|-------------------|
| CALL | TCN CALL, fusao EV_CALL >= EV_PUT, RSI_M5 >= 0.30; EMA/confirm discord = soft Kelly (nao bloqueia) |
| PUT | TCN PUT, fusao EV_PUT > EV_CALL, RSI_M5 <= 0.70; EMA/confirm discord = soft Kelly (nao bloqueia) |
| SKIP tecnico | `training` / `data` / `deploy` / `predict_error`, warm-up, stop-win, broker; `neg_edge` hard se Edge `<= 0`; `neutral_zone`; `neg_edge_zscore_panic`; `live_exec_discord` so se flag on; `anti_loss_seed_discord` so seed unstamped. EMA/confirm/weak/RSI = soft (nao EMPTY) |
| Soft sinal 1.1 | `mini_pair_oppose` / `cal_margin` / loss-clf; chop soft; `neg_edge` soft so se `0 < Edge < min_edge_execute`; EMA/confirm/weak soft; Soft_SIZE piso **2.5%** so se Edge >= **0.015** |
| Fusao multi-escala | `fusion_enabled`: p_eff (Cal + MACRO/janela ops/MINI/MILI/tape + loss continuo + meta **0.0**); `fusion_loss_weight` **0.0** so com `auto=1` — **nao** incorpora o FLIP do mesmo ciclo; `fusion_block_when_tcn_pos_edge` **true** preserva TCN so se Cal **e** raw +EV; `fusion_block_when_tcn_candle_agree` **false** (fusao livre quando janela ops N=3==TCN); telemetria `[GATES] \|\| FUSION` via `fusion_side` + `fusion_ev_*` / `fusion_p_eff` (realinhado ao EXEC apos flip). **M5 last-bar = log; confirmacao = janela N=3.** |
| Flip loss-clf | Apos fusao: `p_loss >= hard_p_loss_floor` (**0.90**) e `veto_ready` + `flip_require_auto_learn` (**true**: seed so SOFT; `p_ovr`/`seed_discord` nao FLIP); **bloqueia FLIP** se Edge Cal **e** raw_edge do TCN >= floor **e** tape/janela ops confirmam TCN (`FLIP_BLOCK:tcn_edge`; tape ou janela ≠ TCN → `flip_waive_tcn_pos_edge_on_discord`); Cal+/raw− libera; sob seed, janela == TCN bloqueia (`FLIP_BLOCK:seed_candle`; `p_ovr` nao fura); seed edge min **−0.08**; live `flip_waive_edge_min` **−1.0**; janela no alvo floor **0.85** so se TCN fraco |
| Chop soft | ADX &lt; **0.22** e (Hurst ∈ [**0.47**, **0.53**] ou SCALE micro=chop) → soft Kelly **0.55**; log `REGIME \|\| CHOP_SOFT` |
| `majority_votes` | Mais votos PUT ou CALL (tape/mili/RSI vs TCN) → adapta lado (se `fusion_replace_adapt_flip` **false**) |

## Catalogo SKIP tecnico

| Razao | Significado |
|-------|-------------|
| `training` | Treino de sessao em andamento |
| `data` | Buffer/historico insuficiente |
| `deploy` | Checkpoint sem `deploy_ok` |
| `predict_error` | Falha de inferencia |
| `neg_edge` | Com SSOT `neg_edge_hard_skip` **true**: Edge `<= 0` → HARD SKIP (`gate_verdict=HARD_SKIP`). Soft so se `0 < Edge < min_edge_*` (`SOFT_SIZE`). Seed+Cal &lt; `neg_edge_deep_edge_floor` **−0.12** marca `boot_deep` no hard. |
| `neg_edge_zscore_panic` | Veto de pânico bilateral: CALL vetado se $Z < -2.0$ (faca caindo); PUT vetado se $Z > +2.0$ (explosão compradora); telemetria `[GATES]` / `EDGE \|\| NEG_ZSCORE_PANIC` com `Z=` / `side=` / `thr=` (±2.0) |
| `anti_loss_ema_slope` | Slope EMA com deteccao rapida: EMA9 slope (2-pontos, `slope_tol * 0.6`) + EMA21 slope (2-pontos, `slope_tol`); CALL exige $\text{EMA}[-1] \ge \text{EMA}[-2] - \text{tol}$; PUT inverso. Cache de EMA por ciclo. **SOFT** Kelly (`anti_loss_soft` / `SOFT_SIZE`, sem Single-Strike) — nao gera EXEC_EMPTY |
| `anti_loss_ema_trend` | Preco vs EMA9 contrario ao lado. Com `anti_loss_allow_candle_flip` **true** + candle valido + Edge Cal vela >= floor (**0.015** explore / **0.010** recovery) → FLIP (`live_exec_flip_to_candle`, soft); Edge subfloor/≤0 → soft no anchor sem flip (`anti_loss_flip_blocked`); senao **SOFT** Kelly — nao gera EXEC_EMPTY sozinho |
| `anti_loss_rsi_momentum` | Soft Kelly pos-lado: CALL atenuado se $\text{RSI} < \text{rsi\_min}$ (default **0.30**); PUT atenuado se $\text{RSI} > \text{rsi\_max}$ (default **0.70**); nao EMPTY |
| `live_exec_discord` | Veto last-bar vs EXEC so com `anti_loss_live_exec_candle_enabled` **true** (SSOT **false** — confirmação = janela ops N=3 + EMA) |
| `anti_loss_seed_discord` | **Seed** unstamped + `p_loss >= 0.85` + TCN pos_edge: **hard SKIP** se janela ops N=3 nao confirma TCN com corpo >= **0.10**. **Live** stampada: `live_discord_weak` / `live_confirm_weak` / `live_weak_candle` / `live_no_candle` / RSI = **SOFT** Kelly (nao EMPTY) |
| Kelly `EXEC_PAUSE` | `stop_win` / `bankroll_below_stake_min` (sizing; **sem** `kelly_no_edge`) |

## Metricas de processo (negocio)

| Situacao | Esperado |
|----------|----------|
| Edge `<= 0` / `neutral_zone` | `EXEC_EMPTY` = processo ok (nao bug) |
| Soft_SIZE + Edge >= **0.015** | stake ~**2.5%** banca (tambem com PEND); sem Single-Strike |
| Soft_SIZE + Edge subfloor | EXEC no piso **1%** (nao ~2.5%) |
| Soft_SIZE + PEND + Edge>=0.015 + stake≈1% | regressao |
| ALLOW + EXPLORE | Single-Strike ~**5%** |
| RECOVER + PEND | Kelly/EXPLORE + Soft_SIZE se aplicavel + caps (`cover_enabled` false) |

## Catalogo sinal / ML (soft + neg_edge soft)

| Razao | Significado |
|-------|-------------|
| `mini_pair_oppose` | Par MINI unanime ≠ lado executado → **sempre** soft Kelly (`mini_pair_soft_kelly_mult` **0.55**); sem hard SKIP |
| `cal_margin` | `direction_margin` &lt; `min_direction_margin` → soft Kelly; waive com pending material |
| `loss_clf_soft` | Container loss-clf: atenua Kelly; log `LOSS_CLF \|\| SOFT` |
| `loss_clf_flip` | `p_loss >= 0.90` + `veto_ready` + waivers; seed: `seed_candle` / `flip_seed_waive_edge_min` **−0.08**; live: `flip_waive_edge_min` **−1.0** → FLIP (`from→to`/`why` em `[GATES]`); senao `FLIP_BLOCK` (seed_candle/seed/scale/neg_edge/tcn_edge) + soft |
| `regime_chop` | ADX/Hurst (ou SCALE chop) → soft Kelly (`chop_soft_kelly_mult` **0.55**); log `REGIME \|\| CHOP_SOFT` |
| `neg_edge` | Soft so se `0 < Edge < min_edge_execute` (`SOFT_SIZE`, **sem** Single-Strike); hard se Edge `<= 0` com `neg_edge_hard_skip` **true** (SSOT); override **false** reabre soft em Edge `<= 0` |
| `anti_loss_soft` | Soft Kelly (`anti_loss_soft_kelly_mult` **0.55**): slope + live confirm/discord/weak + RSI; EMA/candle discord com flip **so se** Edge vela>=floor → `live_exec_flip_to_candle`; seed so se `anti_loss_hard_skip` **false**; telemetria `[GATES] \|\| ANTI_LOSS` |

Quality gate amplo (RSI/discordance/price zone/SIDE_EQ block) permanece **fora** do codigo. Loss-clf live apos ~8–16 settles mistos (`auto=1`); restart container apos mudar `LOSS_BOOTSTRAP_EXIT_N`.
## Escalas MACRO / MICRO / MINI / MILI

Triplo OHLC + ticks: telemetria, **adaptacao de lado** e soft Kelly — **nunca SKIP por escala**.

| Escala | Fonte | Papel |
|--------|-------|-------|
| MACRO | OHLC **86400 s** | Contexto D1 |
| MICRO | OHLC **300 s** | TCN + last-bar na fita (M5) |
| MINI | OHLC **300 s** | Last-bar anterior+atual + slope (M5) |
| MILI | Tick flow | Fluxo intrabar |

Log: `SCALE || … mi_prev=… mi_cur=… tape=… micro=retract|explos|chop adapted=0|1` e no IND `SCALE: tcn=… tape=… votes=C#/P# …`.  
Adapt: **majority_votes** (TCN/tape/mili/RSI) sem hold Cal; tape sob `raw_extreme`; regimes **retracao** / **explosao** / **mili+tape** (mili+tape **nao** adapta em micro=chop; `adapt_mili_tape_skip_chop`). `adapt_allow_strong_tape` **false**. Kelly `kelly_p_floor` **0.55**; com fusao ancora em `fusion_p_eff` so se Cal TCN ja passou neg_edge; explore piso `neutral_bankroll_pct` **1%** + `explore_stake_scale_floor` **0.40**; `fraction` **0.08**; RECOVER sem cover (`cover_enabled` **false**; piso **1%** + caps); damping stop-win inicio **1.0** / perto-meta **0.50**; teto linear3 **3.5%**; payout **0.85**; stop-win Kelly Single-Strike **4.31%**. Sem `kelly_no_edge` / sem SKIP por escala / **sem** zona cinza.
Contrato Deriv **5 m** (ops fixo); label TCN = **N=1 vela M5** (`quantum_multi_barrier`). Ciclo **120 s**; settle live em 5 min. Sem overlap: ciclo bloqueado com contrato aberto.

## `raw_extreme` (anti-override)

Modo de calibracao `raw_extreme` **substituiu** `tcn_macro_override`: raw extremo **nao** substitui Cal; Kelly usa probabilidade calibrada. Chaves SSOT `tcn_macro_call_override` / `tcn_macro_put_override` so limiam raw extremo — **nao** sao timeframe MACRO.

## Edge Cal (identidade Kelly)

Edge = `p_side * (1 + b) - 1` com **p = Cal** do lado (nunca raw). Com `b=0.85`: breakeven `be ≈ 0.541`; floor `min_edge_execute` **0.04** exige Cal ≳ **0.563**. Cal~0.53 → Edge~−0.02 → HARD SKIP com `neg_edge_hard_skip` **true** (SSOT). Soft Kelly (SOFT_SIZE, sem Single-Strike) se `0 < Edge < floor`. Kelly pode ancorar em `fusion_p_eff` **depois** desse gate. `[CLUSTER]` inclui `raw_edge` + `be` sempre que Edge e exibido; `[GATES]` sob soft/EXEC repete o gap e `verdict=`. So telemetria — raw extremo ≠ Kelly.

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
- Caps Kelly / `max_safe_stake_*` (`max_safe_stake_pct` **0.035**; linear2 **0.03**; linear3 **0.025**)
- `risk_management.soft_recovery.cover_enabled: false` — sem amortizacao em massa de pending
- `risk_management.kelly.neutral_bankroll_pct` / `min_stake_pct` **0.01** — piso Kelly **1%** banca (M5)
- `risk_management.kelly.soft_size_min_stake_pct` **0.025** + `soft_size_min_edge` **0.015** — Soft_SIZE elevado **2.5%** (preservado com PEND)
- `orchestrator.execution.signal_skip.anti_loss_allow_candle_flip` **true** — flip microestrutura em EMA/candle discord **somente** com Edge Cal do lado da vela >= `min_edge_explore`/`min_edge_recovery`
- `orchestrator.execution.side_equilibrium.enabled: true` (soft sizing only)
- `orchestrator.execution.scale_vision` (adaptacao de fita + soft sizing; sem SKIP por escala)
- `orchestrator.execution.signal_skip` (1.1 soft Kelly; sem flip pos-LOSS)

Ver doutrina [`llm-trading-doctrine.md`](llm-trading-doctrine.md), [`deriv-indices-algorithm.md`](deriv-indices-algorithm.md), [`infra-docker.md`](infra-docker.md) e [`engineering-settings-ssot.md`](engineering-settings-ssot.md).
