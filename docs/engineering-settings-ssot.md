# Settings SSOT (`config/settings.json`)

O arquivo `config/settings.json` e a fonte unica de verdade (SSOT) para configuracao operacional do motor continuo Laya no `R_10` M5.

## Secoes Principais

- `anchor` e `symbols`: fixados em `"R_10"` e `["R_10"]`.
- `stream`: `micro_granularity: 300` (M5), `ticks_limit` e `buffer_size` para warmup causal.
- `laya`: `enabled: true`, `model_id: "convaiinnovations/laya"`, `device: "auto"`, `context_candles: 12`, `stake_min: 1.0`, `min_stake_pct: 0.01`, `max_stake_pct: 0.05`.
- `risk_management.params`: `duration: 5`, `duration_unit: "m"`, `payout_estimate`, `stake_min: 1.0`, `min_stake_pct: 0.01`, `max_stake_pct: 0.05`.
- `orchestrator`: `engine_mode: "execute"`, intervalos de reconciliacao e cadencia M5 sem bloqueios ou vetos de horario/regime/stop-win.
- `infra` e `telemetry`: configuracao de Redis, TimescaleDB, MinIO e servidor Prometheus local.
- `trading`: `mode` (`demo` ou `real`) e `deriv_app_id` (lido em conjunto com variaveis de ambiente).

## Sanitizacao

Antes de enviar `settings.json` ao contexto do `LayaDecisionAdapter`, chaves sensiveis (`token`, `api_token`, `pat`, `secret`, `password`, `dsn`, `key`, `app_id`) sao removidas recursivamente por `sanitize_settings_context`.
