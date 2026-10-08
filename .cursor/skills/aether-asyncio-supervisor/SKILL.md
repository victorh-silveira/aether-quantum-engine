---
name: aether-asyncio-supervisor
description: >-
  Orienta supervisao asyncio do motor Aether (epoll, TaskGroup, shield,
  ContextVars, offload, starvation do loop). Use when debugging WS stalls,
  settlement lag, CancelledError, or blocking work on the event loop.
---

# Supervisor asyncio

## Quando aplicar

Stall de heartbeat WS, atraso de settlement, tarefas canceladas no meio de write critico, suspeita de bloqueio sync no loop, ou mudanca de TaskGroup/shutdown.

## Checklist

1. Ler `docs/engineering-python-313-runtime.md` (secao asyncio) + `docs/engineering-orchestrator.md`
2. Confirmar: nenhum PyTorch/Polars/NumPy pesado rodando **dentro** do loop (>1-2 ms sync)
3. Offload via `asyncio.to_thread` / executor; TCN: `predict_symbol_decision_async`
4. Estrutura: `TaskGroup` + `CancelledError` propagado; shutdown gracioso do orquestrador
5. Writes Redis/broker criticos sob `asyncio.shield` quando cancel do ciclo nao pode corromper estado
6. Timers em `_scheduled` vs callbacks em `_ready` — nao misturar busy-wait
7. `ContextVars`: copiar/propagar explicitamente se o worker em thread precisar do contexto
8. Medir com py-spy/austin sob carga; nao adivinhar

## Anti-padroes

- `time.sleep` / CPU bound / CUDA sync no coroutine hot path
- Engolir `CancelledError` sem re-raise
- Assumir que ContextVars fluem para `to_thread` automaticamente
- Criar tasks orfas sem supervisao / sem cancel no shutdown
- Poll apertado sem yield que starveie outras tasks

## Refs no repo

- `docs/engineering-python-313-runtime.md`
- `docs/engineering-orchestrator.md`
- `docs/engineering-architecture-senior.md`
- `app/src/application/services/deep_learning/dl_predict_async.py`
- `.cursor/rules/aether-python-313-runtime.mdc`
- `.cursor/rules/aether-orchestrator.mdc`
- `.cursor/rules/aether-architecture-senior.mdc`

## Skills irmas

`aether-python-313-runtime`, `aether-cycle-debug`, `aether-torch-cuda-infer`, `aether-redis-hiredis`, `aether-architecture-senior`
