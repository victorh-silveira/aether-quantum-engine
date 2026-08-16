---
name: aether-cycle-debug
description: >-
  Depura ciclo do orquestrador Aether (warmup M1, signature 60s, training gate,
  EXEC_EMPTY sistemico, watchdog). Use when the cycle stalls, no trades fire,
  warmup hangs, or signature/boundary issues appear in logs.
---

# Cycle debug

## Checklist

1. Warmup / DATA buffer — MACRO **7200 s** / MICRO **60 s** / MINI **60 s** (`stream_sync_start`)
2. FASE TREINO vs OPERACAO (`mandatory_trade_each_cycle` **false**; `online_training` **false**)
3. Signature boundary **60 s** — ciclo alinhado? Contrato Deriv **5 m** (ops fixo; label N do ultimo `[HORIZON] winner`; micro OHLC **60 s**)
4. `training_gate` / `deploy_gate` / predict_error — EMPTY tecnico esperado
5. `force_trade_every_cycle` permanece **false** (nunca “consertar” EMPTY com force)
6. Watchdog stale tick / reconnect stream

Doc: `docs/engineering-orchestrator.md`
