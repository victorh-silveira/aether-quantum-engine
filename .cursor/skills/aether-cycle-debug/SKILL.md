---
name: aether-cycle-debug
description: >-
  Depura ciclo do orquestrador Aether (warmup, signature 60s, training gate,
  EXEC_EMPTY sistemico, watchdog). Use when the cycle stalls, no trades fire,
  warmup hangs, or signature/boundary issues appear in logs.
---

# Cycle debug

## Checklist

1. Warmup / DATA buffer — MACRO 300s / MICRO 60s / MINI 60s (`stream_sync_start`)
2. FASE TREINO vs OPERACAO
3. Signature boundary 60 s — ciclo alinhado? Contrato Deriv 30 s (hibrido)
4. CLUSTER TF — prefere micro (`M1`); Cal/Edge telemetria
5. SCALE — `tape` / `adapted` / discord; adaptacao de lado sob `raw_extreme` (sem SKIP); sizing em `execution_scale_*`
6. Locks / atomic state — deadlock?
7. Watchdog stale/reconnect cooldown (stale tipico 45 s)
8. Pos-settlement segurando a fronteira? (`post_settlement_is_trading_wait_seconds` 35)

Nao ligar `force_trade` para “destravar”.
Nao tratar `raw_extreme` / `tcn_macro_*_override` como timeframe MACRO.

Doc: `docs/engineering-orchestrator.md`
