# Settings SSOT (pos-purge + FLIP no piso)

Leitura: `app/settings_io.py` + parsers em `domain/config_knobs.py`. Knob novo = settings + `resolve_*` + teste + doc se mudar semantica.

## Blocos vivos (execucao)

| Bloco | Papel |
|-------|--------|
| `orchestrator.execution` | mandatory/force off; settlement; SIDE_EQ soft sizing; `scale_vision` telemetria; sample_size_policy |
| `infra.loss_classifier` | HTTP :8006; `veto_mode` **hard**; `hard_p_loss_floor` **0.55**; `flip_young_p_eff_floor` **0.55**; FLIP so apos auto_learn; `flip_trust_n` **32**; tape so telemetria; **sem** Soft Kelly / HARD SKIP / SKIP novo |
| `infra.meta_classifier` | HTTP :8005; `retrain_min_n` **32** |
| `risk_management` | Kelly Single-Strike 4.31%; `cover_enabled` **false**; piso **1%**; caps L0/L1 |

**Removido:** `orchestrator.execution.signal_skip`, `invert_exec_side`, `scale_vision.fusion_*` / `adapt_*` de lado, familia `flip_*` de guards legados.

## Doutrina fail-closed

- Loss-clf = **FLIP** por `p_eff` >=0.55 (so apos auto_learn); tape so telemetria; nao HARD SKIP
- Nao reabrir quality gate amplo / signal_skip multi-gate / Soft do loss-clf
- `force_trade_every_cycle` / `mandatory_trade_each_cycle` **false**
- Catalogo vivo: [`engineering-indicator-gates.md`](engineering-indicator-gates.md)

Skill: `aether-settings-change`. Testes: `test_doctrine_settings_ssot.py` + `doctrine_invariants.py`.
