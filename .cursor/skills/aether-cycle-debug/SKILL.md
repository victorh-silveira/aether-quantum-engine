---
name: aether-cycle-debug
description: >-
  Depura ciclo do orquestrador Aether (warmup M15, signature 900s, training gate,
  EXEC_EMPTY sistemico, watchdog). Use when the cycle stalls, no trades fire,
  warmup hangs, or signature/boundary issues appear in logs.
---

# Cycle debug

## Checklist

1. Warmup / DATA buffer — MACRO **86400 s** (D1) / MICRO **900 s** (M15) / MINI **900 s** (`stream_sync_start`)
2. FASE TREINO vs OPERACAO (`mandatory_trade_each_cycle` **false**; `online_training` **false**)
3. Signature boundary **900 s** — ciclo alinhado? Contrato Deriv **15 m** (ops fixo M15; label N=1; micro OHLC **900 s**)
4. `training_gate` / `deploy_gate` / predict_error — EMPTY tecnico esperado
5. `force_trade_every_cycle` permanece **false** (nunca “consertar” EMPTY com force)
6. Watchdog stale tick / reconnect stream

Doc: `docs/engineering-orchestrator.md`
