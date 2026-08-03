# Settings SSOT (`config/settings.json`)

Unica fonte de knobs de runtime. Parsers fail-closed em `domain/config_knobs.py` e `resolve_*` / `*_config.py`.

## Blocos principais

| Bloco | Papel |
|-------|-------|
| `symbols` / `anchor` | Universo (`R_10`) |
| `data_handler` | macro/micro granularity, history, buffer |
| `deep_learning` | arch, lookback, labels, calib, gating, deploy |
| `orchestrator` | ciclo, warmup, watchdog, WS |
| `orchestrator.execution` | mandatory/force, price_zone, quality_gate, settlement, SIDE_EQ, sample_size_policy, hard_cal_margin_floor |
| `risk_management` | Kelly, soft_recovery, stop-win, ACC gate, duration contrato |
| `infra` | Redis, Timescale, MinIO, Triton, meta |
| `logging` | level, log_file, quiet_channels |
| `auth` / credenciais | PAT — ver [`deriv-api-aether.md`](deriv-api-aether.md) |

## Regra de knob novo

1. Adicionar chave em `settings.json` com default seguro
2. Expor via `resolve_*` / config tipada (fail-closed se obrigatorio)
3. Teste unitario do default e do branch novo
4. Atualizar doc de engenharia ou medallion/doctrine se mudar semantica operacional
5. Nao hardcodar o mesmo numero em tres lugares

## Valores sensiveis (nao afrouxar sem mandato)

- `force_trade_every_cycle: false`
- `hard_cal_margin_floor` (explore, piso **0.05**)
- `align_rsi_trend: true` (SKIP `rsi_trend_misalign` se RSI/DI contra)
- `min_validation_accuracy_gate` (**0.53**)
- `max_safe_stake_cap` / `max_safe_stake_pct`
- `sample_size_policy.*`

Playbook senior: [`binary-senior-playbook.md`](binary-senior-playbook.md).

Skill: `aether-settings-change`.
