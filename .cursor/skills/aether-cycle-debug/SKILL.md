---
name: aether-cycle-debug
description: >-
  Depura ciclo do orquestrador Aether (warmup M5, signature 300s, training gate,
  EXEC_EMPTY sistemico, watchdog). Use when the cycle stalls, no trades fire,
  warmup hangs, or signature/boundary issues appear in logs.
---

# Cycle debug

## Checklist

1. Warmup / DATA buffer — MACRO **86400 s** (D1 / **365** barras, teto `min(365, history_bars)`; **nao** usar piso micro/`inference_history_bars` no D1) / MICRO **300 s** (M5 / **2000** barras) / MINI **300 s** (`stream_sync_start`). Lean treino micro: fetch inicial = `max(fetch_count, micro_fetch_count, training_history_bars)` (**2000**, nao 500). Se log `DATA: ... g=86400s | N/M velas` oscilar (ex. 366↔63↔1) sem fechar: bug de paginacao/merge — ordem epoch + stop sem progresso.
2. FASE TREINO vs OPERACAO (`mandatory_trade_each_cycle` **false**; `online_training` **false**)
3. Signature boundary **300 s** com ciclo de **120 s** — ciclo alinhado? Contrato Deriv **5 m** (ops fixo M5; label N=1 vela M5; micro OHLC **300 s**)
4. `training_gate` / `deploy_gate` / predict_error / cooldown pos-LOSS / pausa de sessao — EMPTY tecnico esperado (sem `SKIP:NEUTRAL_ZONE`)
5. SCALE last: `adapted=1` (`retract_vs_tcn` / `explos_vs_tcn` / `*_holds`) = SCALE fixa EXEC apos FLIP (nao e SKIP; sem freio de margem Cal)
6. `force_trade_every_cycle` permanece **false** (nunca “consertar” EMPTY com force)
7. Watchdog stale tick (**300 s**) / reconnect stream

Doc: `docs/engineering-orchestrator.md`

