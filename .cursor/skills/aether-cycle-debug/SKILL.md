---
name: aether-cycle-debug
description: >-
  Depura ciclo do orquestrador Aether (warmup M2, signature 120s, training gate,
  EXEC_EMPTY sistemico, watchdog). Use when the cycle stalls, no trades fire,
  warmup hangs, or signature/boundary issues appear in logs.
---

# Cycle debug

## Checklist

1. Warmup / DATA buffer — MACRO **3600 s** / MICRO **120 s** / MINI **120 s** (`stream_sync_start`)
2. FASE TREINO vs OPERACAO (`mandatory_trade_each_cycle` **false**; `online_training` **false**)
3. Signature boundary **120 s** — ciclo alinhado? Contrato Deriv **2 m** (micro OHLC **120 s**)
4. CLUSTER TF — micro **M2**; Cal/Edge telemetria
5. SCALE — `tape` / `adapted` / discord; fusao EV `[GATES] || FUSION`; adaptacao de lado sob `raw_extreme` (sem SKIP); sizing em `execution_scale_*`
6. Locks / atomic state — deadlock?
7. Watchdog stale/reconnect cooldown (`watchdog_stale_tick_seconds` **300**)
8. Pos-settlement segurando a fronteira? (`post_settlement_is_trading_wait_seconds` **90**)
9. Pausa tecnica de sizing? stop-win **3%** / `bankroll_below_stake_min` (`EXEC_PAUSE`) — nao “destravar” com force_trade

Nao ligar `force_trade` para “destravar”.
Nao tratar `raw_extreme` / `tcn_macro_*_override` como timeframe MACRO.

Doc: `docs/engineering-orchestrator.md`
