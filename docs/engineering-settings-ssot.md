# Settings SSOT (`config/settings.json`)

Unica fonte de knobs de runtime. Parsers fail-closed em `domain/config_knobs.py` e `resolve_*` / `*_config.py`.

## Blocos principais

| Bloco | Papel |
|-------|-------|
| `symbols` / `anchor` | Universo unico **`R_10`** (Volatility 10 / Deriv) |
| `data_handler` | MACRO/MICRO/MINI granularity, history, buffer |
| `deep_learning` | arch, lookback, labels, calib (`raw_extreme`), deploy, `sample_weighting`; alvo treino **2000** barras M2 com `train_history_shortfall_ratio` **0.95** (API esgotada ~1980 segue); `bootstrap_max_wait_rounds` **16** |
| `orchestrator` | ciclo, warmup, watchdog, WS |
| `orchestrator.execution` | mandatory/force, **`invert_exec_side`** (experimento: inverte CALL/PUT apos gates), settlement, SIDE_EQ soft, `scale_vision`, `signal_skip`, sample_size_policy |
| `infra.meta_classifier` | HTTP :8005; edge continuo 43D; `online_learn` **true**; `/v1/learn` a cada settle (`retrain_min_n` **2**, piso LGBM); `timeout_seconds` **8** |
| `infra.loss_classifier` | HTTP :8006; `veto_mode` **soft** + banda flip: floor soft **0.65**; `hard_p_loss_floor` **0.90**; `flip_require_auto_learn` **true**; `flip_block_when_tcn_pos_edge` **true** (nao FLIP se Edge TCN >= **0.04**); `flip_waive_scale_above_p_loss` **0.95**; `flip_candle_p_loss_floor` **0.85** (so TCN fraco); `flip_waive_edge_min` **-1.0** (live); `flip_seed_block_against_closed_candle` **true** + `flip_seed_waive_edge_min` **-0.08**; seed/`flip_allow_seed_on_scale_discord`; `flip_waive_on_closed_candle`; soft Kelly **0.55→0.40**; `timeout_seconds` **8** |
| `risk_management` | Kelly, soft_recovery, stop-win, ACC gate, duration contrato |
| `infra` | Redis, Timescale, MinIO, Triton, meta |
| `logging` | level, log_file, quiet_channels |
| `auth` / credenciais | PAT — ver [`deriv-api-aether.md`](deriv-api-aether.md) |

## Knobs novos / sensiveis (vies + dinamica + escalas)

| Knob | Bloco | Nota |
|------|-------|------|
| `sample_weighting.*` | `deep_learning` | class_balance + recency (`recency_half_life_n` default **2000**) |
| `reject_majority_collapse` | `deep_learning.deploy_gate` | rejeita collapse de classe no deploy |
| `max_label_call_frac_bias` | idem | padrao **0.20**; `|pred-0.5|` / `|pred-label|` bastam; `|label-0.5|` exige tambem `min_minority_recall` |
| `min_minority_recall` | idem | padrao **0.25** (via label viesado) |
| `early_stopping_patience` / `min_epochs` | `deep_learning` | **16** / **20** (anti-overfit R_10 M2) |
| `weight_decay` / `tcn.dropout` / `learning_rate` | idem | **0.001** / **0.25** / **0.001** |
| `max_brier` / `soft_max_brier` / `eval_brier_max` | `deploy_gate` / DL | **0.26** (mini alinhado ao soft) |
| `train_deploy_retries` | `deep_learning` | **5** tentativas com reseed ate `deploy_ok` |
| `side_equilibrium.enabled` | `orchestrator.execution` | soft Kelly only; sem veto de direcao |
| `scale_vision.*` | `orchestrator.execution` | `adapt_allow_strong_tape` **false**; **majority_votes** (TCN/tape/mili/RSI + vela micro fechada se `adapt_majority_include_micro_bar` **true**); `adapt_majority_min_lead` **2**; `adapt_skip_chop` **true** (hold TCN em micro=chop); `adapt_require_cal_agree` **true** (nao adapta contra Cal); `adapt_mili_tape_skip_chop` **true**; **fusao EV** `fusion_enabled` **true** + `fusion_replace_adapt_flip` **true** (argmax EV CALL/PUT com pesos MACRO/vela/MINI/MILI/tape/loss/meta; `fusion_meta_ev_weight` **0.10**; `fusion_loss_weight` **0.45**; `fusion_tcn_shrink_near_half` **0.25**; `fusion_block_when_tcn_pos_edge` **true**; `fusion_block_when_tcn_candle_agree` **true** (`why=tcn_candle_agree`); `fusion_loss_requires_auto_learn` **true** + `fusion_loss_seed_weight_mult` **0.0** (seed nao alimenta loss_bonus); soft Kelly se EV escolhido &lt; `fusion_min_edge_execute` via `fusion_weak_ev_soft_kelly_mult` **0.40**; sob seed e ambos EV &lt; 0 → `fusion_weak_ev_seed_soft_kelly_mult` **0.25**; log `[GATES] \|\| FUSION`); **sem** `adapt_*_cal_margin` / hold cinza |
| `signal_skip.*` | `orchestrator.execution` | Escopo **1.1**: mini/cal/chop soft Kelly **0.55**; **neg_edge_hard_skip** **false**; soft continuo com `neg_edge_soft_min_edge` (**-1.0**); sob seed `neg_edge_bootstrap_soft_kelly_mult` **0.25** + hard so se edge &lt; `neg_edge_deep_edge_floor` (**-0.12**); Edge = `Cal*(1+b)-1`; floor **0.04** exige Cal ≳ **0.605** para edge positivo |
| `scale_vision.adapt_on_majority_votes` | idem | Conta votos TCN/tape/mili/mini_pair/RSI; lideranca ≥`adapt_majority_min_lead` e n≥`adapt_majority_min_votes` → `majority_votes` |
| `kelly.kelly_p_floor` | `risk_management.kelly` | Piso de **probabilidade** para Kelly; garante `f*>0`; alias `adapt_kelly_p_floor`; com `fusion_applied`, Kelly ancora em `fusion_p_eff` do lado escolhido |
| `kelly.target_damping_floor` / `target_damping_span` | `risk_management.kelly` | Damping stop-win: inicio sessao **1.0** (`floor` **0.50** + `span` **0.50`); perto da meta **0.50** (cover RECOVER nao esmagado no arranque) |
| `sample_size_policy.explore_stake_scale_floor` | `orchestrator.execution` | Piso relativo EXPLORE cold-start (**0.40**); doutrina exige `>0` |
| `kelly.neutral_bankroll_pct` | `risk_management.kelly` | Piso operacional de stake explore (**0.25%** banca M2); loss_clf soft **nao** esmaga o piso |
| `kelly.payout_fallback` / `params.payout_estimate` / `default_payout` | `risk_management` | Payout Deriv R_10 M2 **0.72** (live; cover RECOVER = `cover_multiple * pending/0.72`) |
| `kelly.stop_win_kelly_*` | `risk_management.kelly` | Boost stop-win ~**1h**: `enabled`, `cycles_target` **4**, `live_n_min` **0**, fracoes **0.70–1.0**, teto **5%** |
| `soft_recovery.infeasible_force_explore` | `risk_management.soft_recovery` | Default **true**: `RECOVERY_INFEASIBLE` ou cover≥cap → EXPLORE Kelly (sem DAL no teto) |
| `soft_recovery.pending_waives_scale_explore` | `risk_management.soft_recovery` | Default **true**: pending material libera soft cover apesar de `scale_adapted`/`scale_force_explore` |
| `soft_recovery.adapted_force_explore` | `risk_management.soft_recovery` | Default **true**: `scale_adapted` + linear≥**2** → EXPLORE (bloqueia DAL L2/L3 sob adapt) |
| `soft_recovery.cover_multiple` | `risk_management.soft_recovery` | Multiplo do cover (**1.50**) — amortiza pending sem progressao geometrica |
| `soft_recovery.max_safe_stake_pct` | `risk_management.soft_recovery` | Teto RECOVER **5%** banca; linear2 **4%**; linear3+ **2.5%** |
| `kelly.recovery_min_val_accuracy` | `risk_management.kelly` | Piso ACC live para DAL (**0.53**); sobe com linear; abaixo → EXPLORE (sem cover DAL) |
| `soft_recovery.live_evidence_force_explore_*` | `risk_management.soft_recovery` | linear≥**3** + `live_n`≥**2** + `live_wr`&lt;**0.62** → EXPLORE (bloqueia DAL L3+ com ACC de treino ainda ok) |
| `soft_recovery.amort_cycles_min` / `amort_cycles_max` | `risk_management.soft_recovery` | Amort **2–4**; stake RECOVER = `cover_multiple * pending/payout/amort` × damping (sem `max` com progressao exponencial; `f*` so gate) |
| `infra.loss_classifier.soft_max_stake_pct_high` | `infra.loss_classifier` | Teto stake EXPLORE sob soft (**0.25%**); waivado com pending material; ACC baixo nao cancela cover |
| `params.duration` | `risk_management.params` | Contrato RISE_FALL **2 m** (`duration_unit: m`) — universo `R_10` M2 |
| `data_handler.micro_granularity` / `granularity` | `data_handler` | Micro/MINI **120** / macro **3600** (M2) |
| `deep_learning.lookback` | `deep_learning` | **720** barras micro @ **120 s** |
| `orchestrator.cycle_interval_seconds` / `signature_boundary_seconds` | `orchestrator` | **120 s** (alinhado ao fecho da vela M2 / contrato 2 m); `exec_empty_retry` **120 s** |
| `orchestrator.settlement_tolerance_window_seconds` | `orchestrator` | **90** (contrato 2 m) |
| `orchestrator.watchdog_stale_tick_seconds` | `orchestrator` | **300** |
| `orchestrator.post_settlement_is_trading_wait_seconds` | `orchestrator` | **90** |
| `tcn_macro_call_override` / `tcn_macro_put_override` | `deep_learning.calibration` | limiar de **raw** para modo `raw_extreme`; Cal nao e substituido |
| `calibration_neutral_drift` | `deep_learning.calibration` | banda neutra efetiva **[0.47, 0.53]**; drift degenerado `[0.5,0.5]` e rejeitado (cai em `neutral_half_width`) |
| `calibration.method` | `deep_learning.calibration` | **auto** (Brier/ECE com piso de sharpness; fallback `identity`) |
| `mini_granularity` | `data_handler` | padrao **120** (MINI OHLC M2) |

Removidos: `decision_threshold_call` / `decision_threshold_put` (mortos). Modo `tcn_macro_override` (substituir Cal por raw) removido — usar `raw_extreme`. Removidos: `adapt_min_cal_margin` / `adapt_max_cal_margin` / `hold_calib_gray` / `hold_cal_margin` / `calib_gray_*` / log `CALIB_GRAY`.

## Regra de knob novo

1. Adicionar chave em `settings.json` com default seguro
2. Expor via `resolve_*` / config tipada (fail-closed se obrigatorio)
3. Teste unitario do default e do branch novo
4. Atualizar doc de engenharia ou medallion/doctrine se mudar semantica operacional
5. Nao hardcodar o mesmo numero em tres lugares

## Valores sensiveis (nao afrouxar sem mandato)

- `force_trade_every_cycle: false`
- `min_validation_accuracy_gate` (**0.53**)
- `max_safe_stake_cap` / `max_safe_stake_pct`
- `sample_size_policy.*`

Vetos de sinal/qualidade amplos (RSI/cal floor/quality_gate/price_zone/SIDE_EQ block) permanecem **fora** (escopo 1). Flip loss-clf (`hard_p_loss_floor`) permanece sob mandato **2026-08-07** (+ `flip_waive_on_closed_candle`, `flip_waive_edge_min` **-1.0** live; seed: `flip_seed_block_against_closed_candle` + `flip_seed_waive_edge_min` **-0.08**); chop e **neg_edge** = soft Kelly continuo (`neg_edge_hard_skip` **false**), com hard bootstrap profundo (`neg_edge_deep_edge_floor`). SIDE_EQ restante = soft Kelly sizing. Apos mudar env do loss-clf: **restart** `aether-loss-classifier`.

Playbook senior: [`binary-senior-playbook.md`](binary-senior-playbook.md).

Skill: `aether-settings-change`.
