---
name: aether-python-deps
description: >-
  Audita e altera requirements Python do Aether (anti-redundancia, Polars-only,
  ABI numpy/torch/sklearn). Use when adding or removing pip packages, editing
  requirements*.txt, or resolving dependency conflicts.
---

# Python deps

## Ordem

1. Ler `docs/engineering-python-deps.md`
2. Grep `import`/`from` do pacote no first-party (`app/`, `infra/docker/`)
3. Se so-transitivo de wrapper (ex. `coverage` via `pytest-cov`): nao declarar
4. Se import direto: declarar no requirements do ambiente certo
5. DataFrame: **somente Polars**; nunca `pandas` / `to_pandas` / dual-stack / 3a lib DF
6. LightGBM: passar NumPy (`frame.to_numpy()`), nao DataFrame cru
7. Atualizar `app/requirements.txt` e/ou `requirements-dev.txt` e/ou Docker reqs
8. WSL: `python -m pip check`
9. Se mexeu em numpy/torch/sklearn: smoke `import numpy,torch,sklearn,joblib,polars`
10. Rodar pre-commit / testes de politica `test_python_deps_policy`

## Anti-padroes

Pinar `coverage` junto de `pytest-cov`; reintroduzir `pandas`; remover `joblib` com imports vivos; ignorar falha de ABI apos bump.
