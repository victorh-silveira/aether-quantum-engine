# Settings SSOT (`config/settings.json`)

Unica fonte de knobs de runtime. Parsers fail-closed em `domain/config_knobs.py` e `resolve_*` / `*_config.py`.

## Blocos principais

| Bloco | Papel |
|-------|-------|
| `symbols` / `anchor` | Universo unico **`OTC_SPC`** (S&P 500 OTC; nao Volatility) |
| `data_handler` | MACRO/MICRO/MINI granularity, history, buffer |
| `deep_learning` | arch, lookback, labels, calib (`raw_extreme`), deploy, `sample_weighting` |
| `orchestrator` | ciclo, warmup, watchdog, WS |
| `orchestrator.execution` | mandatory/force, settlement, SIDE_EQ soft, `scale_vision`, `signal_skip`, sample_size_policy |
| `infra.meta_classifier` | HTTP :8005; edge continuo 43D |
| `infra.loss_classifier` | HTTP :8006; `veto_mode` **soft** (atenua Kelly; log `LOSS_CLF \|\| SOFT`); floor **0.65**; `soft_kelly_mult` **0.55** → `soft_kelly_mult_high` **0.40** @ `soft_p_loss_high` **0.85**; teto stake EXPLORE `soft_max_stake_pct_high` **2%** (waivado com pending para cover); piso neutral so escala sob loss_clf soft; `ready_n`/`retrain_min_n` **24**; `retrain_on_loss_min_n` **2**; `timeout_seconds` **8**; buffer persistido |
| `risk_management` | Kelly, soft_recovery, stop-win, ACC gate, duration contrato |
| `infra` | Redis, Timescale, MinIO, Triton, meta |
| `logging` | level, log_file, quiet_channels |
| `auth` / credenciais | PAT — ver [`deriv-api-aether.md`](deriv-api-aether.md) |

## Knobs novos / sensiveis (vies + dinamica + escalas)

| Knob | Bloco | Nota |
|------|-------|------|
| `sample_weighting.*` | `deep_learning` | class_balance + recency (`recency_half_life_n` default 2000) |
| `reject_majority_collapse` | `deep_learning.deploy_gate` | rejeita collapse de classe no deploy |
| `max_label_call_frac_bias` | idem | padrao **0.20** |
| `min_minority_recall` | idem | padrao **0.25** |
| `side_equilibrium.enabled` | `orchestrator.execution` | soft Kelly only; sem veto de direcao |
| `scale_vision.*` | `orchestrator.execution` | `adapt_allow_strong_tape` **false**; **majority_votes** (TCN/tape/mili/RSI); explosion/mili/retract; **sem** `adapt_*_cal_margin` / hold cinza |
| `signal_skip.*` | `orchestrator.execution` | Escopo **1.1**: mini/cal soft Kelly **0.55**; **sem** `calib_gray_*`; **sem** flip pos-LOSS; **sem** hard SKIP de sinal |
| `scale_vision.adapt_on_majority_votes` | idem | Conta votos TCN/tape/mili/mini_pair/RSI; lideranca ≥`adapt_majority_min_lead` e n≥`adapt_majority_min_votes` → `majority_votes` |
| `kelly.kelly_p_floor` | `risk_management.kelly` | Piso de **probabilidade** para Kelly; garante `f*>0`; alias `adapt_kelly_p_floor` |
| `kelly.neutral_bankroll_pct` | `risk_management.kelly` | Piso operacional de stake explore (**2%** banca); loss_clf soft **nao** esmaga o piso |
| `kelly.payout_fallback` / `params.payout_estimate` / `default_payout` | `risk_management` | Payout Deriv OTC_SPC M15 **0.72** (live; cover RECOVER = `cover_multiple * pending/0.72`) |
| `kelly.stop_win_kelly_*` | `risk_management.kelly` | Boost stop-win ~**1h**: `enabled`, `cycles_target` **4**, `live_n_min` **0**, fracoes **0.70–1.0**, teto **5%** |
| `soft_recovery.infeasible_force_explore` | `risk_management.soft_recovery` | Default **true**: `RECOVERY_INFEASIBLE` ou cover≥cap → EXPLORE Kelly (sem DAL no teto) |
| `soft_recovery.pending_waives_scale_explore` | `risk_management.soft_recovery` | Default **true**: pending material libera soft cover apesar de `scale_adapted`/`scale_force_explore` |
| `soft_recovery.adapted_force_explore` | `risk_management.soft_recovery` | Default **true**: `scale_adapted` + linear≥**2** → EXPLORE (bloqueia DAL L2/L3 sob adapt) |
| `soft_recovery.cover_multiple` | `risk_management.soft_recovery` | Multiplo do cover (**2.0**) — WIN zera pending e deixa lucro ~pending |
| `soft_recovery.max_safe_stake_pct` | `risk_management.soft_recovery` | Teto RECOVER **5%** banca (linear2/3 tambem **5%**) |
| `kelly.recovery_min_val_accuracy` | `risk_management.kelly` | Piso ACC live para DAL (**0.53**); sobe com linear; abaixo → EXPLORE (sem cover DAL) |
| `soft_recovery.live_evidence_force_explore_*` | `risk_management.soft_recovery` | linear≥**3** + `live_n`≥**2** + `live_wr`&lt;**0.58** → EXPLORE (bloqueia DAL L3+ com ACC de treino ainda ok) |
| `soft_recovery.amort_cycles_min` / `amort_cycles_max` | `risk_management.soft_recovery` | Cover em 1 ciclo (`amort=1`); stake = `cover_multiple * pending/payout` |
| `infra.loss_classifier.soft_max_stake_pct_high` | `infra.loss_classifier` | Teto stake EXPLORE sob soft (**2%**); waivado com pending material; ACC baixo nao cancela cover |
| `params.duration` | `risk_management.params` | Contrato RISE_FALL **15 m** (`duration_unit: m`) — universo `OTC_SPC` somente M15 |
| `data_handler.micro_granularity` / `granularity` | `data_handler` | Micro/MINI **900** / macro **3600** (1:5; M15) |
| `deep_learning.lookback` | `deep_learning` | **720** barras micro (~7,5 dias @ 900 s) |
| `orchestrator.cycle_interval_seconds` / `signature_boundary_seconds` | `orchestrator` | **15 s** (entrada continua; nao espera fronteira M15); `exec_empty_retry` **15 s** |
| `orchestrator.settlement_tolerance_window_seconds` | `orchestrator` | **300** (contrato 15 m) |
| `orchestrator.watchdog_stale_tick_seconds` | `orchestrator` | **600** (OTC SPX quieto) |
| `tcn_macro_call_override` / `tcn_macro_put_override` | `deep_learning.calibration` | limiar de **raw** para modo `raw_extreme`; Cal nao e substituido |
| `calibration.method` | `deep_learning.calibration` | **auto** (Brier/ECE com piso de sharpness; fallback `identity`) |
| `mini_granularity` | `data_handler` | padrao **60** (MINI OHLC) |

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

Vetos de sinal/qualidade (Hurst/ADX/RSI/cal floor/quality_gate/price_zone/SIDE_EQ block) foram **removidos do codigo** (mandato escopo 1). SIDE_EQ restante = soft Kelly sizing.

Playbook senior: [`binary-senior-playbook.md`](binary-senior-playbook.md).

Skill: `aether-settings-change`.
