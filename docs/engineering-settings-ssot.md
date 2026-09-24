# Settings SSOT (pos-purge + FLIP no piso)

Politica de mercado atual: `orchestrator.execution.four_market_vetoes=true`
substitui os knobs individuais legados de mercado; `market_direction_trigger=true`
habilita reavaliacao anterior ao edge, sem alterar probabilidades do TCN.
Ambos exigem booleano `true` (default desligado). Piso economico compartilhado:
`min_edge_execute`. Contrato e limitacoes: [catalogo](engineering-indicator-gates.md).

Leitura: `app/settings_io.py` + parsers em `domain/config_knobs.py`. Knob novo = settings + `resolve_*` + teste + doc se mudar semantica.

## Blocos vivos (execucao)

| Bloco | Papel |
|-------|--------|
| `orchestrator.execution` | mandatory/force off; `invert_exec_side` **false**; confluencia senior nao e knob direcional; `skip_neg_edge` **true** sem smart waive em EXPLORE; `skip_exec_vs_candle` / `skip_scale_candle_discord` / `skip_doji` **false**; `skip_below_soft_min_acc` **false**; settlement; SIDE_EQ soft sizing; `scale_vision` (adapt off); sample_size_policy; `counter_trend_min_edge` **0.08** |
| `orchestrator.execution.meta_payoff` | LEARN/telemetria; soft Kelly **inerte** (nao comprime stake) |
| `orchestrator.execution.scale_vision` | `adapt_retract_enabled` **false**; knobs de retract/tape/explos permanecem no JSON mas nao viram lado |
| `infra.loss_classifier` | HTTP :8006; `veto_mode` **hard**; `hard_p_loss_floor` **0.70**; `flip_young_p_eff_floor` **0.58**; FLIP so apos auto_learn; `flip_min_n_train` **1**; `flip_trust_n` **64**; `flip_young_shrink` **0.50**; `ready_n` **32**; `retrain_min_n` / `retrain_on_loss_min_n` **12** e `min_win_for_loss_retrain` **4**; o seed e apenas telemetria ate haver dados reais das duas classes; **sem** Soft Kelly / HARD SKIP |
| `infra.meta_classifier` | HTTP :8005; `retrain_min_n` **32** |
| `risk_management` | Kelly Single-Strike 4.31% so com `live_n >= 12` e conviction ≥ **0.58**; `cover_enabled` **true** (`cover_multiple` **1.0**); `amort_cycles_min/max` **1/1**; stake recovery = `min(max(PEND/payout, 1% banca), cap_L0)`; cover usa L0 **3.5%** (`cover_l0`, ignora L2/L3); a primeira proposta valida atualiza o payout liquido somente para a sessao e o edge/BE dos ciclos seguintes; PEND nao force-explore por near-stop; piso **1%** soberano; caps EXPLORE L2/L3 |

**Removido:** `orchestrator.execution.signal_skip`, `senior_confluence_flip`, `cal_soft_edge_*`, `scale_vision.fusion_*` / `adapt_direction_enabled` legado amplo, familia `flip_*` de guards legados.

## Doutrina fail-closed

- TCN live: sempre CALL se Cal ≥**0.5** senao PUT; `confidence_call_threshold` **0.57** / `confidence_put_threshold` **0.43** nao skipam; `calibration_neutral_drift` **[0.45, 0.55]** / `neutral_half_width` / `min_calibration_margin_floor` **0.05** (live: raw se Cal mole e raw nitido; sem stretch); `apply_calibrator_stable` prefere raw se mais nitido; `temperature_min` **0.75**; `force_ok` **true** so exporta para diagnostico; execucao exige ACC anti-colapso >=**0.50**, Brier settlement **<0.260**, minimo **120** e LCB Wilson 90% >= breakeven do payout, alem de anti-collapse
- `deep_learning.deploy_gate.provisional_*`: modo controlado quando taxa OOS >=**0.57**, N>=**120** e Brier <**0.260**, mas LCB Wilson pleno ainda falha; stake soberanamente limitada a **1%** da banca, sem waiver por recovery/PEND
- Loss-clf = **FLIP** young `p_eff` >=0.58 / mature >=0.70 (so apos auto_learn; `flip_young_shrink` **0.50**); tape so telemetria; nao HARD SKIP
- Nao reabrir quality gate amplo / signal_skip multi-gate / Soft do loss-clf
- `force_trade_every_cycle` / `mandatory_trade_each_cycle` **false**
- Catalogo vivo: [`engineering-indicator-gates.md`](engineering-indicator-gates.md)

Skill: `aether-settings-change`. Testes: `test_doctrine_settings_ssot.py` + `doctrine_invariants.py`.
