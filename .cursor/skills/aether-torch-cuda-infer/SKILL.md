---
name: aether-torch-cuda-infer
description: >-
  Orienta inferencia Torch/CUDA no host Aether (inference_mode, to_thread,
  batch 1, pinned memory, compile offline). Use when changing TCN predict,
  CUDA stalls, event-loop blocking, or torch.compile experiments.
---

# Inferencia Torch / CUDA (host)

## Quando aplicar

Mudanca em predicao TCN, stall de ciclo sob CUDA, bloqueio do event loop, pinned memory, ou experimento `torch.compile`.

## Checklist

1. Ler `docs/engineering-python-313-runtime.md` + `docs/engineering-deep-learning.md` + `docs/engineering-architecture-senior.md`
2. Inferencia critica no **host** (nao no container); batch **1**; eager local
3. Contrato async: `app/src/application/services/deep_learning/dl_predict_async.py` — `predict_symbol_decision_async` chama `await asyncio.to_thread(eager_local_predict, ...)`
4. Forward sob `torch.inference_mode()` (preferivel a so `no_grad`)
5. Nunca rodar forward CUDA sincronamente no loop asyncio
6. Pinned memory / transfers: medir; nao assumir ganho sem evidencia no WSL
7. `torch.compile`: **opt-in offline** (treino/bench); nao default no hot path live
8. Meta/loss permanecem HTTP sidecars `:8005`/`:8006` com timeout/fallback
9. Checkpoint lookback/granularity/horizon alinhados ao SSOT settings

## Anti-padroes

- Forward Torch dentro da coroutine sem `to_thread`
- `torch.compile` ligado por default em DEMO/prod sem mandato
- Batch >1 no path live de decisao
- Importar `torch` em `domain/`
- Disputar VRAM com Optuna/tuning enquanto o motor live roda

## Refs no repo

- `app/src/application/services/deep_learning/dl_predict_async.py`
- `app/src/application/services/deep_learning/dl_predict_build.py` (`eager_local_predict`)
- `docs/engineering-python-313-runtime.md`
- `docs/engineering-deep-learning.md`
- `docs/engineering-architecture-senior.md`
- `.cursor/rules/aether-deep-learning.mdc`
- `.cursor/rules/aether-python-313-runtime.mdc`
- `.cursor/rules/aether-architecture-senior.mdc`

## Skills irmas

`aether-asyncio-supervisor`, `aether-dl-train`, `aether-python-313-runtime`, `aether-polars-arrow`, `aether-architecture-senior`
