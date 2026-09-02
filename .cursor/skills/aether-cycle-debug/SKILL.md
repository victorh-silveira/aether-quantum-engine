---
name: aether-cycle-debug
description: >-
  Depura ciclo do orquestrador Aether (warmup M5, signature 300s, training gate,
  EXEC_EMPTY sistemico, watchdog). Use when the cycle stalls, no trades fire,
  warmup hangs, or signature/boundary issues appear in logs.
---

# Cycle debug

## Checklist

1. Warmup / DATA buffer — MACRO **86400 s** (D1 / 365 barras) / MICRO **300 s** (M5 / 500 barras) / MINI **300 s** (`stream_sync_start`)
2. FASE TREINO vs OPERACAO (`mandatory_trade_each_cycle` **false**; `online_training` **false**)
3. Signature boundary **300 s** com ciclo de **120 s** — ciclo alinhado? Contrato Deriv **5 m** (ops fixo M5; label N=1 vela M5; micro OHLC **300 s**)
4. `training_gate` / `deploy_gate` / `neutral_zone` / predict_error — EMPTY tecnico esperado
5. `force_trade_every_cycle` permanece **false** (nunca “consertar” EMPTY com force)
6. Watchdog stale tick (**300 s**) / reconnect stream

Doc: `docs/engineering-orchestrator.md`

