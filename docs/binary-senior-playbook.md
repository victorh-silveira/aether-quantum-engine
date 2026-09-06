# Playbook trader senior — binarias M5 (`1HZ75V`; OHLC 300s)

Postura operacional (**escopo 1.1** + arquitetura continua 1HZ75V): TCN **14D** ancora Cal (**0.53/0.47**); **fusao EV** multi-escala escolhe CALL/PUT; **loss-clf**; **micro_protect**; depois **regime boolean** (`regime_squeeze` HARD sem flip); anti-loss direcional **off** no SSOT. SKIP tecnico = treino/dados/deploy/broker/stop-win/regime/micro; **neg_edge** HARD se Edge `<= 0` ou Edge `< 0.015`; Soft_SIZE so soft flags com Edge >= floor.

Universo: **Volatility 75 (1s) Index** (`1HZ75V`) — **M5** (contrato ops **5 m / M5**; label TCN **N=1** vela M5; ciclo **300 s** com `require_signature_boundary` **true**; micro/MINI **300 s**, macro **86400 s**, ratio **1:288**; meta **23D**).

Hierarquia: TCN Cal/Margin → SCALE dirs → soft `signal_skip` → **fusao EV** → **loss-clf FLIP** → **micro_protect** → **regime boolean** (`regime_gate_enabled` **true**) → chop soft → **neg_edge** → Kelly Single-Strike / caps. **Proibido** reabrir quality gate amplo legado; **proibido** narrar flip anti-loss como filtro vivo com flags off.

## Quando operar

| Lado | Condicoes tipicas |
|------|-------------------|
| CALL | TCN CALL (Cal >= **0.53**), fusao EV_CALL >= EV_PUT, Edge >= **0.015**, regime operable |
| PUT | TCN PUT (Cal <= **0.47**), fusao EV_PUT > EV_CALL, Edge >= **0.015**, regime operable |
| SKIP tecnico | `training` / `data` / `deploy` / `predict_error`, warm-up, stop-win, broker; `regime_squeeze`; `neg_edge` hard se Edge `<= 0` ou Edge `< 0.015`; `neutral_zone`; `neg_edge_zscore_panic` |
| Soft sinal 1.1 | `mini_pair_oppose` / `cal_margin` / loss-clf; chop soft; Soft_SIZE so soft flags **com** Edge >= **0.015**; Soft_SIZE piso **2.5%** so se Edge >= **0.015** |
| Fusao multi-escala | `fusion_enabled`: p_eff (Cal + MACRO/janela ops/MINI/MILI/tape + loss continuo + meta **0.0**); `fusion_loss_weight` **0.0** so com `auto=1` — **nao** incorpora o FLIP do mesmo ciclo; `fusion_block_when_tcn_pos_edge` **true** preserva TCN so se Cal **e** raw +EV; `fusion_block_when_tcn_candle_agree` **false** (fusao livre quando janela ops N=3==TCN); telemetria `[GATES] \|\| FUSION` via `fusion_side` + `fusion_ev_*` / `fusion_p_eff` (realinhado ao EXEC apos flip). **M5 last-bar = log; confirmacao = janela N=3.** |
| Flip loss-clf | Apos fusao: `p_loss >=` floor efetivo (`hard_p_loss_floor` **0.90**, ou `flip_waive_guards_above_p_loss` **0.85** sob override) e `veto_ready`; seed so SOFT **exceto** candle-discord ou `flip_waive_guards_above_p_loss`; com p_loss >= **0.85** fura `tcn_edge`/`seed`/`seed_candle` (`why=p_ovr`); pos-FLIP: micro nao desfaz; Edge Cal fraco → Soft_SIZE (`neg_edge_flip_candle`) mesmo se vela ≠ EXEC |
| Chop soft | ADX &lt; **0.22** e (Hurst ∈ [**0.47**, **0.53**] ou SCALE micro=chop) → soft Kelly **0.55**; log `REGIME \|\| CHOP_SOFT` |
| `majority_votes` | Mais votos PUT ou CALL (tape/mili/RSI vs TCN) → adapta lado (se `fusion_replace_adapt_flip` **false**) |

## Catalogo SKIP tecnico

| Razao | Significado |
|-------|-------------|
| `training` | Treino de sessao em andamento |
| `data` | Buffer/historico insuficiente |
| `deploy` | Checkpoint sem `deploy_ok` |
| `predict_error` | Falha de inferencia |
| `neg_edge` | Com SSOT `neg_edge_hard_skip` **true**: Edge `<= 0` → HARD SKIP; Edge `< min_edge_*` (**0.015** explore = recovery, inclusive subfloor positivo) → HARD (`neg_edge_subfloor_hard`, `gate_verdict=HARD_SKIP`). Soft_SIZE **nao** vem de subfloor. Excecao: apos `loss_clf_flip` com vela == EXEC e `neg_edge_soft_when_closed_candle_agree` **true** → Soft_SIZE (`neg_edge_flip_candle`) em vez de HARD. Seed+Cal &lt; `neg_edge_deep_edge_floor` **−0.12** marca `boot_deep` no hard. |
| `neg_edge_zscore_panic` | Veto de pânico bilateral: CALL vetado se $Z < -2.0$ (faca caindo); PUT vetado se $Z > +2.0$ (explosão compradora); telemetria `[GATES]` / `EDGE \|\| NEG_ZSCORE_PANIC` com `Z=` / `side=` / `thr=` (±2.0) |
| `regime_squeeze` | HARD SKIP: ADX &lt; `regime_adx_max` (**0.1**) e BB squeeze (`regime_bb_squeeze_enabled` **true**); **nao** altera CALL/PUT; telemetria `[GATES] \|\| REGIME` |
| `micro_discord` | HARD SKIP so se follow falhar: vela M5 ≠ EXEC com corpo >= `micro_discord_min_body` (**0.1**) e Edge Cal do lado da vela &lt; `min_edge_*` (**0.015**); com `micro_discord_follow_candle` **true** e Edge vela >= piso → FOLLOW Soft_SIZE (`why=micro_discord_follow`, mult **0.55**); telemetria `micro_discord_confirmed` / `[GATES] \|\| MICRO FOLLOW|HARD` |
| `chop_loss_risk` | HARD SKIP: soft/FLIP_BLOCK + `loss_clf_p_loss` >= **0.90** + vela M5 ≠ EXEC; **nao** flip; vela alinhada ou ausente → Soft_SIZE segue |
| `soft_confirm_weak` | HARD SKIP: soft/FLIP_BLOCK + `confirm_score` < `soft_exec_min_confirmations` (**2**) entre peers definidos (vela/tape/mi/mili/ops); **nao** flip |
| Kelly `EXEC_PAUSE` | `stop_win` / `bankroll_below_stake_min` (sizing; **sem** `kelly_no_edge`) |

## Metricas de processo (negocio)

| Situacao | Esperado |
|----------|----------|
| Edge `<= 0` / Edge `< floor` / `neutral_zone` | `EXEC_EMPTY` = processo ok (nao bug); subfloor → `neg_edge_subfloor_hard` |
| Soft_SIZE + Edge >= **0.015** | stake ~**2.5%** banca (tambem com PEND); sem Single-Strike |
| Soft_SIZE com Edge `< floor` | regressao (deveria HARD; Soft_SIZE so soft flags com Edge >= floor) |
| Soft_SIZE + PEND + Edge>=0.015 + stake≈1% | regressao |
| ALLOW + EXPLORE | Single-Strike ~**5%** |
| RECOVER + PEND | Kelly/EXPLORE + Soft_SIZE se aplicavel + caps (`cover_enabled` false) |

## Catalogo sinal / ML (soft + neg_edge soft)

| Razao | Significado |
|-------|-------------|
| `mini_pair_oppose` | Par MINI unanime ≠ lado executado → **sempre** soft Kelly (`mini_pair_soft_kelly_mult` **0.55**); sem hard SKIP |
| `cal_margin` | `direction_margin` &lt; `min_direction_margin` → soft Kelly; waive com pending material |
| `loss_clf_soft` | Container loss-clf: atenua Kelly; stamp `gate_verdict=SOFT_SIZE` tambem com PEND (`SOFT_WAIVE_PENDING`); log `LOSS_CLF \|\| SOFT` |
| `loss_clf_flip` | `p_loss >= 0.90` + `veto_ready` + waivers; seed: `seed_candle` / `flip_seed_waive_edge_min` **−0.08**; live: `flip_waive_edge_min` **−1.0** → FLIP (`from→to`/`why` em `[GATES]`); senao `FLIP_BLOCK` (seed_candle/seed/scale/neg_edge/tcn_edge) + soft |
| `regime_chop` | ADX/Hurst (ou SCALE chop) → soft Kelly (`chop_soft_kelly_mult` **0.55**); log `REGIME \|\| CHOP_SOFT` |
| `neg_edge` | Hard se Edge `<= 0` ou Edge `< min_edge_*` (**0.015** explore = recovery) com `neg_edge_hard_skip` **true** (SSOT; `neg_edge_subfloor_hard` no subfloor positivo); Soft_SIZE **nao** para `0 < Edge < floor`; override **false** reabre soft em Edge `<= 0` / subfloor |
| `anti_loss_soft` | Soft Kelly (`anti_loss_soft_kelly_mult` **0.55**): slope + live confirm/discord/weak + RSI; hybrid `anti_loss_anchor_agree=false` + last ≠ EXEC → `live_discord_weak` (ou flip se Edge last>=floor); EMA/candle discord com flip **so se** Edge vela>=floor → `live_exec_flip_to_candle`; seed so se `anti_loss_hard_skip` **false**; telemetria `[GATES] \|\| ANTI_LOSS` |

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
Contrato Deriv **5 m** (ops fixo); label TCN = **N=1 vela M5** (`quantum_multi_barrier`). Ciclo **300 s** (`require_signature_boundary` **true**, abertura M5); settle live em 5 min. Sem overlap: ciclo bloqueado com contrato aberto.

## `raw_extreme` (anti-override)

Modo de calibracao `raw_extreme` **substituiu** `tcn_macro_override`: raw extremo **nao** substitui Cal; Kelly usa probabilidade calibrada. Chaves SSOT `tcn_macro_call_override` / `tcn_macro_put_override` so limiam raw extremo — **nao** sao timeframe MACRO.

## Edge Cal (identidade Kelly)

Edge = `p_side * (1 + b) - 1` com **p = Cal** do lado (nunca raw). Com `b=0.85`: breakeven `be ≈ 0.541`; floor `min_edge_explore`/`min_edge_recovery` **0.015** exige Cal ≳ **0.549**. Cal~0.53 → Edge~−0.02 → HARD SKIP com `neg_edge_hard_skip` **true** (SSOT). Subfloor positivo (`0 < Edge < floor`) tambem HARD (`neg_edge_subfloor_hard`) — **nao** Soft_SIZE. Soft_SIZE so soft flags com Edge >= floor. Kelly pode ancorar em `fusion_p_eff` **depois** desse gate. `[CLUSTER]` inclui `raw_edge` + `be` sempre que Edge e exibido; `[GATES]` sob soft/EXEC repete o gap e `verdict=`. So telemetria — raw extremo ≠ Kelly.

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
- `risk_management.kelly.soft_size_min_stake_pct` **0.025** + `soft_size_min_edge` **0.015** — Soft_SIZE elevado **2.5%** (preservado com PEND e com D-SQUEEZE via `d_squeeze_floor_waived_for_soft_size`)
- `orchestrator.execution.signal_skip.anti_loss_allow_candle_flip` **false** — flip microestrutura em EMA/candle discord **somente** com Edge Cal do lado da vela >= `min_edge_explore`/`min_edge_recovery`
- `orchestrator.execution.side_equilibrium.enabled: true` (soft sizing only)
- `orchestrator.execution.scale_vision` (adaptacao de fita + soft sizing; sem SKIP por escala)
- `orchestrator.execution.signal_skip` (1.1 soft Kelly; sem flip pos-LOSS)

Ver doutrina [`llm-trading-doctrine.md`](llm-trading-doctrine.md), [`deriv-indices-algorithm.md`](deriv-indices-algorithm.md), [`infra-docker.md`](infra-docker.md) e [`engineering-settings-ssot.md`](engineering-settings-ssot.md).
