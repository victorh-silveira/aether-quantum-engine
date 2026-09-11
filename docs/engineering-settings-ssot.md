# Settings SSOT (pos-purge + FLIP no piso)

Leitura: `app/settings_io.py` + parsers em `domain/config_knobs.py`. Knob novo = settings + `resolve_*` + teste + doc se mudar semantica.

## Blocos vivos (execucao)

| Bloco | Papel |
|-------|--------|
| `orchestrator.execution` | mandatory/force off; settlement; SIDE_EQ soft sizing; `scale_vision` (telemetria + adapt retract); sample_size_policy; `post_loss_cooldown` (LIN>=1: 300/300/600/900s) |
| `infra.loss_classifier` | HTTP :8006; `veto_mode` **hard**; `hard_p_loss_floor` **0.58**; `flip_young_p_eff_floor` **0.55**; FLIP so apos auto_learn; `flip_trust_n` **32**; `ready_n` **32**; `retrain_min_n` **12**; saida do seed live **4** (`bootstrap_exit_n` / `LOSS_BOOTSTRAP_EXIT_N`, nao `ready_n`); `retrain_on_loss_min_n` / `min_win_for_loss_retrain` **4**; tape so telemetria; **sem** Soft Kelly / HARD SKIP |
| `orchestrator.execution.scale_vision` | `adapt_retract_enabled` **true**; `retraction_require_mili` **true** (retract **e** explos exigem mili); `adapt_tape_require_strong` **true** (discordance so telemetria); last apos FLIP (sem `adapt_soft_margin`) |
| `orchestrator.execution.meta_payoff_veto` | soft Kelly: edge ≤ 0 → `soft_veto_score_factor` **0.85**; edge ≤ **`soft_veto_strong_edge` -0.5** → `soft_veto_strong_factor` **0.40**; sat=1+Cal≤0 também soft; **nao** re-eleva stake com `soft_size_min_stake_pct`; sem SKIP |
| `infra.meta_classifier` | HTTP :8005; `retrain_min_n` **32** |
| `risk_management` | Kelly Single-Strike 4.31% so com `live_n >= 12` e conviction ≥ **0.58**; `cover_enabled` **false**; piso **1%**; caps L0/L1 |

**Removido:** `orchestrator.execution.signal_skip`, `invert_exec_side`, `cal_soft_edge_*`, `scale_vision.fusion_*` / `adapt_direction_enabled` legado amplo, familia `flip_*` de guards legados.

## Doutrina fail-closed

- TCN live: sempre CALL se Cal ≥**0.5** senao PUT; `confidence_call_threshold` **0.55** / `confidence_put_threshold` **0.45** nao skipam; `calibration_neutral_drift` **[0.45, 0.55]** / `neutral_half_width` / `min_calibration_margin_floor` **0.05** so telemetria e ramo `raw_extreme`; `apply_calibrator_stable` prefere raw se mais nitido; `temperature_min` **0.75**; sharpness **0.03**; gap **0.05**; ACC/`eval_*` **0.53** inalterado
- Loss-clf = **FLIP** young `p_eff` >=0.55 / mature >=0.58 (so apos auto_learn); tape so telemetria; nao HARD SKIP
- Nao reabrir quality gate amplo / signal_skip multi-gate / Soft do loss-clf
- `force_trade_every_cycle` / `mandatory_trade_each_cycle` **false**
- Catalogo vivo: [`engineering-indicator-gates.md`](engineering-indicator-gates.md)

Skill: `aether-settings-change`. Testes: `test_doctrine_settings_ssot.py` + `doctrine_invariants.py`.
