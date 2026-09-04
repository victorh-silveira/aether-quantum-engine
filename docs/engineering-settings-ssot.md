# Settings SSOT (`config/settings.json`)

Unica fonte de knobs de runtime. Parsers fail-closed em `domain/config_knobs.py` e `resolve_*` / `*_config.py`.

## Blocos principais

| Bloco | Papel |
|-------|-------|
| `symbols` / `anchor` | Universo unico **`1HZ75V`** (Volatility 75 (1s) Index / Deriv) |
| `data_handler` | MACRO/MICRO/MINI granularity: M5 (**300 s**), macro D1 (**86400 s**), history 365 barras |
| `deep_learning` | arch, lookback, labels, calib (`raw_extreme`), deploy, `sample_weighting`; treino em 365 velas diarias |
| `orchestrator` | ciclo (**120 s**), signature boundary (**300 s**), warmup, watchdog, WS |
| `orchestrator.execution` | mandatory/force, **`invert_exec_side`** (experimento: inverte CALL/PUT apos gates), settlement, SIDE_EQ soft, `scale_vision`, `signal_skip`, sample_size_policy |
| `infra.meta_classifier` | HTTP :8005; edge continuo 43D; `online_learn` **true**; `/v1/learn` a cada settle (`retrain_min_n` **2**, piso LGBM); `timeout_seconds` **8** |
| `infra.loss_classifier` | HTTP :8006; `veto_mode` **soft** + banda flip: floor soft **0.65**; `hard_p_loss_floor` **0.90**; `flip_require_auto_learn` **true**; soft Kelly **0.55→0.40**; `timeout_seconds` **8** |
| `risk_management` | Kelly Single-Strike (**4.31% da banca em 1 trade M5**, payout **0.85**), soft_recovery, stop-win **4.31%**, duration contrato **5 m** |
| `infra` | Redis, Timescale, MinIO, meta, loss |
| `logging` | level, log_file, quiet_channels |
| `auth` / credenciais | PAT — ver [`deriv-api-aether.md`](deriv-api-aether.md) |

## Knobs novos / sensiveis (vies + dinamica + escalas)

| Knob | Bloco | Nota |
|------|-------|------|
| `sample_weighting.*` | `deep_learning` | class_balance + recency (`recency_half_life_n` default **500**) |
| `reject_majority_collapse` | `deep_learning.deploy_gate` | rejeita collapse de classe no deploy |
| `max_label_call_frac_bias` | idem | padrao **0.20**; `|pred-0.5|` / `|pred-label|` bastam; `|label-0.5|` exige tambem `min_minority_recall` |
| `min_minority_recall` | idem | padrao **0.25** (via label viesado) |
| `early_stopping_patience` / `min_epochs` | `deep_learning` | **25** / **15** (anti-overfit 1HZ75V M5) |
| `weight_decay` / `tcn.dropout` / `learning_rate` | idem | **0.005** / **0.35** / **0.001** |
| `max_brier` / `soft_max_brier` / `eval_brier_max` | `deploy_gate` / DL | **0.28** (mini alinhado ao soft) |
| `train_deploy_retries` | `deep_learning` | **6** tentativas com reseed ate `deploy_ok` |
| `side_equilibrium.enabled` | `orchestrator.execution` | soft Kelly only; sem veto de direcao |
| `scale_vision.*` | `orchestrator.execution` | `ops_window_bars` **3** (confirmacao = deslocamento liquido das ultimas 3 velas M5 fechadas; M5 last-bar = log); `adapt_allow_strong_tape` **false**; **majority_votes** (TCN/tape/mili/RSI + vela micro fechada se `adapt_majority_include_micro_bar` **true**); `adapt_majority_min_lead` **2**; `adapt_skip_chop` **true** (hold TCN em micro=chop); `adapt_require_cal_agree` **true** (nao adapta contra Cal); `adapt_mili_tape_skip_chop` **true**; soft sizing discord: `kelly_mult_discord` **0.55** + `max_stake_pct_discord` **0.05** (nao capar Single-Strike sob mini-pair); **fusao EV** `fusion_enabled` **true** + `fusion_replace_adapt_flip` **true** (argmax EV CALL/PUT com pesos MACRO/janela ops/MINI/MILI/tape/loss/meta; `fusion_meta_ev_weight` **0.0**; `fusion_loss_weight` **0.0**; `fusion_tcn_shrink_near_half` **0.25**; `fusion_block_when_tcn_pos_edge` **true**; `fusion_block_when_tcn_candle_agree` **false**; `fusion_loss_requires_auto_learn` **true** + `fusion_loss_seed_weight_mult` **0.0** (seed nao alimenta loss_bonus); soft Kelly se EV escolhido &lt; `fusion_min_edge_execute` via `fusion_weak_ev_soft_kelly_mult` **0.50**; sob seed e ambos EV &lt; 0 → `fusion_weak_ev_seed_soft_kelly_mult` **0.25**; log `[GATES] \|\| FUSION`); **sem** `adapt_*_cal_margin` / hold cinza |
| `signal_skip.*` | `orchestrator.execution` | Escopo **1.1**: mini/cal/chop soft Kelly **0.55** (`chop_pause_enabled` **false**); `min_direction_margin` **0.005**; **neg_edge_hard_skip** **true** → Edge `<= 0` HARD SKIP (`gate_verdict=HARD_SKIP`); soft SOFT_SIZE so se `0 < Edge < min_edge_*` (**sem** Single-Strike; `neg_edge_soft_min_edge` **-1.0**); seed soft mult **0.25** + `boot_deep` se Cal &lt; `neg_edge_deep_edge_floor` **-0.12** no hard; Z-score panic hard com telemetria `Z=`/`side`/`thr`; Edge = `Cal*(1+b)-1`; floor **0.04** exige Cal ≳ **0.605** para edge positivo; **anti-loss** (`anti_loss_hard_skip` **true**): ancora hibrida (`anti_loss_anchor_mode=hybrid`: janela ops N=3 + ultima vela micro fechada; telemetria `anti_loss_ops_dir`, `anti_loss_last_dir`, `anti_loss_anchor_agree`); cache EMA por ciclo (`invalidate_ema_cache` no inicio do ciclo; `calc_ema_series` em `execution_anti_loss_helpers.py`); EMA slope/trend + live confirm/discord/weak + RSI **soft Kelly** (preco vs EMA9; RSI **0.30/0.70** pos-lado; `anti_loss_soft_kelly_mult` **0.55**); seed unstamped HARD; seed unstamped+discord com `p_loss` floor **0.85** + pos_edge; live com **janela ops N=3** stampada (mesmo `auto=0`): tier fraco (corpo liquido &lt; **0.10**); `anti_loss_live_exec_candle_enabled` **false** (sem HARD `live_exec_discord` por last-bar); gate unificado (`anti_loss_live_confirm_enabled` **true**): soft se a janela nao confirma o **lado EXEC** com `anti_loss_live_confirm_min_body` **0.15** no corpo liquido das 3 M5 (`anti_loss_soft_kelly_mult` **0.55**) |
| `scale_vision.adapt_on_majority_votes` | idem | Conta votos TCN/tape/mili/mini_pair/RSI; lideranca ≥`adapt_majority_min_lead` e n≥`adapt_majority_min_votes` → `majority_votes` |
| `kelly.kelly_p_floor` | `risk_management.kelly` | Piso de **probabilidade** para Kelly; garante `f*>0`; alias `adapt_kelly_p_floor`; com `fusion_applied`, Kelly ancora em `fusion_p_eff` do lado escolhido **so se** Cal TCN ja passou `neg_edge` |
| `kelly.target_damping_floor` / `target_damping_span` | `risk_management.kelly` | Damping stop-win: inicio sessao **1.0** (`floor` **0.50** + `span` **0.50`); perto da meta **0.50** (cover RECOVER nao esmagado no arranque) |
| `sample_size_policy.explore_stake_scale_floor` | `orchestrator.execution` | Piso relativo EXPLORE cold-start (**0.40**); doutrina exige `>0` |
| `kelly.neutral_bankroll_pct` / `min_stake_pct` / `micro_bankroll_pct` | `risk_management.kelly` | Piso operacional de stake Kelly (**1%** banca M5); proibido comprimir abaixo |
| `kelly.soft_size_min_stake_pct` / `soft_size_max_stake_pct` / `soft_size_min_edge` | `risk_management.kelly` | Piso Soft_SIZE (**2.5%** banca) so se Edge >= **0.015**; aplica tambem com PEND (`cover_enabled` false); subfloor permanece no piso **1%**; sem Single-Strike |
| `signal_skip.anti_loss_allow_candle_flip` | `orchestrator.execution.signal_skip` | **true** — EMA/candle discord → FLIP para lado da vela/ops (soft) **so se** Edge Cal vela >= `min_edge_explore`/`min_edge_recovery`; subfloor → soft no anchor; ao flipar sync `fusion_p_eff`; Edge final ≤0 → HARD neg_edge |
| `kelly.payout_fallback` / `params.payout_estimate` / `default_payout` | `risk_management` | Payout Deriv **1HZ75V** **0.85** (live) |
| `kelly.mandatory_weak_max_stake_pct` | `risk_management.kelly` | Cap mandatory weak alinhado ao piso (**1%**) |
| `kelly.stop_win_kelly_*` | `risk_management.kelly` | Boost stop-win Single-Strike: `enabled`, `cycles_target` **1**, `live_n_min` **0**, fracoes **1.0–1.0**, teto **5%** |
| `soft_recovery.infeasible_force_explore` | `risk_management.soft_recovery` | Com PEND material, cover≥cap → stake=**CAP** (parcial); flag so afeta ramo legado sem passivo material |
| `soft_recovery.pending_waives_scale_explore` | `risk_management.soft_recovery` | Default **true**: pending material libera soft cover apesar de `scale_adapted`/`scale_force_explore` |
| `soft_recovery.adapted_force_explore` | `risk_management.soft_recovery` | Default **true**: `scale_adapted` + linear≥**2** → EXPLORE (bloqueia DAL L2/L3 sob adapt) |
| `soft_recovery.cover_enabled` | `risk_management.soft_recovery` | **false** — sem sizing por cover amortizado; PEND so telemetria/ledger |
| `soft_recovery.cover_multiple` | `risk_management.soft_recovery` | Legado (**1.10**) — inerte com `cover_enabled` false |
| `soft_recovery.amort_cycles_min` / `amort_cycles_max` | `risk_management.soft_recovery` | Legado amort **2/3** — inerte com `cover_enabled` false; caps `max_safe_stake_*` ativos |
| `soft_recovery.linear_bankroll_pct` | `risk_management.soft_recovery` | Unidade linear alinhada ao piso (**1%**) |
| `soft_recovery.max_safe_stake_pct` | `risk_management.soft_recovery` | Teto RECOVER L0/L1 **3.5%** banca; linear2 **3.0%**; linear3+ **2.5%** |
| `kelly.recovery_min_val_accuracy` | `risk_management.kelly` | Piso ACC live para DAL (**0.53**); sobe com linear; abaixo → EXPLORE (sem cover DAL) |
| `soft_recovery.live_evidence_force_explore_*` | `risk_management.soft_recovery` | linear≥**3** + `live_n`≥**2** + `live_wr`&lt;**0.62** → EXPLORE (bloqueia DAL L3+ com ACC de treino ainda ok) |
| `infra.loss_classifier.soft_max_stake_pct_high` | `infra.loss_classifier` | Teto stake EXPLORE sob soft (**5%**, alinhado a Single-Strike / `max_stake_pct`); waivado com pending material e com `FLIP_BLOCK` (keep TCN so atenua f*) |
| `params.duration` | `risk_management.params` | Contrato RISE_FALL **5 m** (`duration_unit: m`) — fixo via `horizon_sweep.ops_contract_duration_minutes` (**5**); promote **nao** exporta duration do winner |
| `deep_learning.label_horizon_bars` | `deep_learning` | **N=1** vela M5 do TCN (`quantum_multi_barrier`); alinhado ao contrato ops **5 m** |
| `horizon_sweep.*` | `deep_learning` | Grade **`n_bars=[1,2,3,4]`** / **`duration_minutes=[5,10,15,20]`** (celulas H{N} no relogio M5); `ops_contract_duration_minutes` **5**; `quiet_train_logs` **true** (celula **CRITICAL** + `why=` se deploy=0; pos-sweep denso); `run_in_launch_train` **true**; pisos settle be+0.03, n≥16, history≥800 |
| `data_handler.micro_granularity` / `granularity` | `data_handler` | Micro/MINI **300** / macro **86400** (M5/D1; ratio **1:288**) |
| `deep_learning.lookback` | `deep_learning` | **30** barras micro @ **300 s** (tensor `[1, 30, 34]`) |
| `orchestrator.cycle_interval_seconds` / `signature_boundary_seconds` | `orchestrator` | **120 s** / **300 s** (alinhado ao fecho da vela M5); `exec_empty_retry` **120 s** |
| `orchestrator.settlement_tolerance_window_seconds` | `orchestrator` | **600** (slack pos-expiry; `doctrine_invariants` exige **600**) |
| `orchestrator.watchdog_stale_tick_seconds` | `orchestrator` | **300** |
| `orchestrator.post_settlement_is_trading_wait_seconds` | `orchestrator` | **90** |
| `tcn_macro_call_override` / `tcn_macro_put_override` | `deep_learning.calibration` | limiar de **raw** para modo `raw_extreme`; Cal nao e substituido |
| `calibration_neutral_drift` | `deep_learning.calibration` | banda neutra efetiva **[0.47, 0.53]**; drift degenerado `[0.5,0.5]` e rejeitado (cai em `neutral_half_width`) |
| `calibration.method` | `deep_learning.calibration` | **auto** (Brier/ECE com piso de sharpness; fallback `identity`) |
| `min_calibration_margin_floor` | `deep_learning.calibration` | **0.03** — restaura raw se Cal colapsar para dentro da banda neutra |
| `confidence_call_threshold` / `confidence_put_threshold` | `deep_learning` | **0.53** / **0.47** (gray zone = banda ±3pp) |
| `max_calibrated_raw_gap` | `deep_learning.calibration` | **0.08** — clipa p_call **antes** da zona neutra (`cal_raw_gap_capped`) |
| `temperature_min` | `deep_learning.calibration` | **1.0** — impede afiar logits (T&lt;1) no fit |
| `mini_granularity` | `data_handler` | padrao **300** (MINI OHLC M5) |

Removidos: `decision_threshold_call` / `decision_threshold_put` (mortos). Modo `tcn_macro_override` (substituir Cal por raw) removido — usar `raw_extreme`. Removidos: `adapt_min_cal_margin` / `adapt_max_cal_margin` / `hold_calib_gray` / `hold_cal_margin` / `calib_gray_*` / log `CALIB_GRAY`. Removidos (higiene): bloco top-level `gating` / `risk` (nao confundir com `risk_management` nem `snapshot["risk"]`); `api_config.ws_connect_*` flat (usar `orchestrator.ws_connect` / `api_config.ws_connect`); `data_handler.history_fetch_*` (usar `api_config.history_fetch`).

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

Vetos de sinal/qualidade amplos (RSI/cal floor/quality_gate/price_zone/SIDE_EQ block) permanecem **fora** (escopo 1). Flip loss-clf (`hard_p_loss_floor`) permanece sob mandato **2026-08-07** (+ `flip_waive_on_closed_candle`, `flip_waive_edge_min` **-1.0** live; seed: `flip_seed_block_against_closed_candle` + `flip_seed_waive_edge_min` **-0.08**); chop = soft Kelly continuo; **neg_edge** com `neg_edge_hard_skip` **true**: Edge `<= 0` HARD; soft SOFT_SIZE so no subfloor positivo (bloqueia Single-Strike); bootstrap profundo (`neg_edge_deep_edge_floor`) no hard. SIDE_EQ restante = soft Kelly sizing. Apos mudar env do loss-clf: **restart** `aether-loss-classifier`.

Playbook senior: [`binary-senior-playbook.md`](binary-senior-playbook.md).

Skill: `aether-settings-change`.
