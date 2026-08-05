# Settings SSOT (`config/settings.json`)

Unica fonte de knobs de runtime. Parsers fail-closed em `domain/config_knobs.py` e `resolve_*` / `*_config.py`.

## Blocos principais

| Bloco | Papel |
|-------|-------|
| `symbols` / `anchor` | Universo (`R_10`) |
| `data_handler` | MACRO/MICRO/MINI granularity, history, buffer |
| `deep_learning` | arch, lookback, labels, calib (`raw_extreme`), deploy, `sample_weighting` |
| `orchestrator` | ciclo, warmup, watchdog, WS |
| `orchestrator.execution` | mandatory/force, settlement, SIDE_EQ soft, `scale_vision`, sample_size_policy |
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
| `scale_vision.*` | `orchestrator.execution` | `adapt_on_retraction` / `adapt_on_explosion` / `adapt_on_mili_tape`; micro=explos/retract/chop; sem SKIP |
| `kelly.kelly_p_floor` | `risk_management.kelly` | Piso de **probabilidade** para Kelly; garante `f*>0`; alias `adapt_kelly_p_floor` |
| `kelly.neutral_bankroll_pct` | `risk_management.kelly` | Piso operacional de stake explore (**0.25%** banca; contrato 30s / ciclo 60s) quando Kelly puro fica subcentavo; evita EXEC em `$1` broker |
| `soft_recovery.infeasible_force_explore` | `risk_management.soft_recovery` | Default **true**: `RECOVERY_INFEASIBLE` ou cover≥cap → EXPLORE Kelly (sem DAL no teto) |
| `soft_recovery.pending_waives_scale_explore` | `risk_management.soft_recovery` | Default **true**: pending material libera soft cover apesar de `scale_adapted`/`scale_force_explore` |
| `soft_recovery.max_safe_stake_pct` | `risk_management.soft_recovery` | Teto RECOVER **5%** banca (mandato); linear2/3 dampen via knobs dedicados |
| `soft_recovery.amort_cycles_min` / `amort_cycles_max` | `risk_management.soft_recovery` | Janela de amortizacao do cover `pending/payout` (**2–5**; alinhado medallion) |
| `params.duration` | `risk_management.params` | Contrato RISE_FALL **30 s**; ciclo/micro OHLC **60 s** (híbrido) |
| `data_handler.micro_granularity` / `granularity` | `data_handler` | Micro **60** / macro **300** (1:5) |
| `deep_learning.lookback` | `deep_learning` | **720** barras micro (~12 h @ 60 s) |
| `tcn_macro_call_override` / `tcn_macro_put_override` | `deep_learning.calibration` | limiar de **raw** para modo `raw_extreme`; Cal nao e substituido |
| `calibration.method` | `deep_learning.calibration` | **auto** (Brier/ECE com piso de sharpness; fallback `identity`) |
| `mini_granularity` | `data_handler` | padrao **60** (MINI OHLC) |

Removidos: `decision_threshold_call` / `decision_threshold_put` (mortos). Modo `tcn_macro_override` (substituir Cal por raw) removido — usar `raw_extreme`.

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
