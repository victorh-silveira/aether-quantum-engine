---
name: aether-cycle-debug
description: >-
  Depura ciclo do orquestrador Aether (warmup, signature 120s, training gate,
  EXEC_EMPTY sistemico, watchdog). Use when the cycle stalls, no trades fire,
  warmup hangs, or signature/boundary issues appear in logs.
---

# Cycle debug

## Checklist

1. Warmup / DATA buffer — simbolos e barras micro/macro
2. FASE TREINO vs OPERACAO
3. Signature boundary 120 s — ciclo alinhado?
4. CLUSTER Cal/Edge — EXEC_EMPTY por gate e processo valido?
5. Locks / atomic state — deadlock?
6. Watchdog stale/reconnect cooldown
7. Pos-settlement segurando a fronteira?

Nao ligar `force_trade` para “destravar”.

Doc: `docs/engineering-orchestrator.md`
