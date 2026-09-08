# Settings SSOT (pos-purge gates)

Leitura: `app/settings_io.py` + parsers em `domain/config_knobs.py`. Knob novo = settings + `resolve_*` + teste + doc se mudar semantica.

## Blocos vivos (execucao)

| Bloco | Papel |
|-------|--------|
| `orchestrator.execution` | mandatory/force off; settlement; SIDE_EQ soft sizing; `scale_vision` telemetria; sample_size_policy |
| `infra.loss_classifier` | HTTP :8006; `veto_mode` **hard**; `hard_p_loss_floor` **0.90**; `ready_n` / retrain / buffer; **sem** `flip_*` / soft Kelly |
| `infra.meta_classifier` | HTTP :8005; `retrain_min_n` **32** |
| `risk_management` | Kelly Single-Strike 4.31%; `cover_enabled` **false**; piso **1%**; caps L0/L1 |

**Removido:** `orchestrator.execution.signal_skip`, `invert_exec_side`, `scale_vision.fusion_*` / `adapt_*` de lado, familia `flip_*` do loss-clf.

## Doutrina fail-closed

- Loss-clf so HARD SKIP por `P_LOSS` (`gate_reason=loss_clf`)
- Nao reabrir quality gate amplo / signal_skip multi-gate / FLIP
- `force_trade_every_cycle` / `mandatory_trade_each_cycle` **false**
- Catalogo vivo: [`engineering-indicator-gates.md`](engineering-indicator-gates.md)

Skill: `aether-settings-change`. Testes: `test_doctrine_settings_ssot.py` + `doctrine_invariants.py`.
