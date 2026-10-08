---
name: aether-polars-arrow
description: >-
  Audita DataFrame SSOT Polars/Arrow no Aether (LazyFrame, zero-copy,
  POLARS_MAX_THREADS, banimento de pandas). Use when touching features,
  tabular pipelines, to_numpy borders, or suspected pandas imports.
---

# Polars / Arrow SSOT

## Quando aplicar

Features DL, pipelines tabulares, borda tensorial `to_numpy`, mudanca de `POLARS_MAX_THREADS`, suspeita de `pandas`/`to_pandas`, ou review de PR de dados.

## Checklist

1. Ler `docs/engineering-python-313-runtime.md` + `docs/engineering-python-deps.md`
2. Unica lib DF: **Polars**; preferir **LazyFrame** ate a borda de collect
3. Collect/transform pesado: fora do loop asyncio (`to_thread`) ou custo <1-2 ms
4. Arrow → NumPy/torch: caminho zero-copy quando layout permitir; sem `to_pandas`
5. Grep fail-closed: `pandas`, `to_pandas`, `modin`, `cudf`, `dask.dataframe`
6. DF **nao** entra em `domain/` como entidade; dominio recebe valores/arrays tipados
7. `POLARS_MAX_THREADS` consciente do loop e da VRAM/CUDA no host
8. Pins alinhados a `app/requirements.txt` (skill `aether-python-deps`)

## Anti-padroes

- Importar ou reintroduzir `pandas`
- Dual-stack Polars+pandas “temporario”
- `collect()` enorme no hot path do ciclo sem offload
- Tratar DataFrame como objeto de dominio
- Ignorar contiguity e forcar copia silenciosa na borda torch

## Refs no repo

- `docs/engineering-python-313-runtime.md`
- `docs/engineering-python-deps.md`
- `docs/engineering-architecture-senior.md`
- `docs/engineering-deep-learning.md`
- `app/requirements.txt`
- `.cursor/rules/aether-python-deps.mdc`
- `.cursor/rules/aether-python-313-runtime.mdc`
- `.cursor/rules/aether-domain-pure.mdc`

## Skills irmas

`aether-python-313-runtime`, `aether-python-deps`, `aether-dl-train`, `aether-torch-cuda-infer`, `aether-architecture-senior`
