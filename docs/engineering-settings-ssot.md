# Settings SSOT (pos-purge + FLIP no piso)

Leitura: `app/settings_io.py` + parsers em `domain/config_knobs.py`. Knob novo = settings + `resolve_*` + teste + doc se mudar semantica.

## Blocos vivos (execucao)

| Bloco | Papel |
|-------|--------|
| `orchestrator.execution` | mandatory/force off; settlement; SIDE_EQ soft sizing; `scale_vision` telemetria; sample_size_policy; `post_loss_cooldown` (LIN>=1: 300/300/600/900s) |
| `infra.loss_classifier` | HTTP :8006; `veto_mode` **hard**; `hard_p_loss_floor` **0.58**; `flip_young_p_eff_floor` **0.55**; FLIP so apos auto_learn; `flip_trust_n` **32**; `ready_n` **32**; `retrain_min_n` **12**; saida do seed live **12** (`LOSS_BOOTSTRAP_EXIT_N`, nao `ready_n`); `retrain_on_loss_min_n` / `min_win_for_loss_retrain` **4**; tape so telemetria; **sem** Soft Kelly / HARD SKIP / SKIP novo |
| `infra.meta_classifier` | HTTP :8005; `retrain_min_n` **32** |
| `risk_management` | Kelly Single-Strike 4.31% so com `live_n >= 12` e conviction ≥ **0.58**; `cover_enabled` **false**; piso **1%**; caps L0/L1 |

**Removido:** `orchestrator.execution.signal_skip`, `invert_exec_side`, `scale_vision.fusion_*` / `adapt_*` de lado, familia `flip_*` de guards legados.

## Doutrina fail-closed

- TCN live: sempre CALL se Cal ≥**0.5** senao PUT; `confidence_call_threshold` **0.55** / `confidence_put_threshold` **0.45** nao skipam; `calibration_neutral_drift` **[0.45, 0.55]** / `neutral_half_width` / `min_calibration_margin_floor` **0.05** so telemetria e ramo `raw_extreme`; `apply_calibrator_stable` prefere raw se mais nitido; `temperature_min` **0.75**; sharpness **0.03**; gap **0.05**; ACC/`eval_*` **0.53** inalterado
- Loss-clf = **FLIP** young `p_eff` >=0.55 / mature >=0.58 (so apos auto_learn); tape so telemetria; nao HARD SKIP
- Nao reabrir quality gate amplo / signal_skip multi-gate / Soft do loss-clf
- `force_trade_every_cycle` / `mandatory_trade_each_cycle` **false**
- Catalogo vivo: [`engineering-indicator-gates.md`](engineering-indicator-gates.md)

Skill: `aether-settings-change`. Testes: `test_doctrine_settings_ssot.py` + `doctrine_invariants.py`.
