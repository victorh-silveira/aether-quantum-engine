---
name: aether-python-deps
description: >-
  Audita e altera requirements Python do Aether (anti-redundancia, Polars-only,
  ABI numpy/torch/sklearn, httpx/websockets/MinIO hot-path). Use when adding or
  removing pip packages, editing requirements*.txt, or resolving dependency conflicts.
---

# Python deps

## Ordem

1. Ler `docs/engineering-python-deps.md`
2. Grep `import`/`from` do pacote no first-party (`app/`, `infra/docker/`)
3. Se so-transitivo de wrapper (ex. `coverage` via `pytest-cov`): nao declarar
4. Se import direto: declarar no requirements do ambiente certo
5. DataFrame: **somente Polars**; nunca `pandas` / `to_pandas`; LazyFrame preferido; `POLARS_MAX_THREADS` no bootstrap
6. Network: httpx singleton (pools meta/loss); websockets com `max_size`/`ping_*` via `apply_websocket_connect_defaults`
7. MinIO: apenas via `asyncio.to_thread`; LightGBM: NumPy na borda + threads limitadas
8. Atualizar `app/requirements.txt` e/ou `requirements-dev.txt` e/ou Docker reqs
9. WSL: `python -m pip check`
10. Se mexeu em numpy/torch/sklearn: smoke `import numpy,torch,sklearn,joblib,polars`
11. Rodar pre-commit / `test_python_deps_policy` + testes WS de defaults

## Anti-padroes

- Pinar `coverage` junto de `pytest-cov`; reintroduzir `pandas`; remover `joblib` com imports vivos
- `httpx.AsyncClient()` por sinal; omitir `max_size`/`ping_*` no WSS
- MinIO sync na corrotina; Polars pesado no thread do loop; LightGBM `n_jobs=-1` no sidecar
- Ignorar falha de ABI apos bump

## Skills irmas

`aether-polars-arrow`, `aether-torch-cuda-infer`, `aether-redis-hiredis`, `aether-asyncpg-timescale`, `aether-deriv-connect`, `aether-python-313-runtime`

Doc: `docs/engineering-python-deps.md` + `docs/engineering-python-313-runtime.md` + `docs/engineering-architecture-senior.md`
