---
name: aether-python-deps
description: >-
  Audita e altera requirements Python do Aether (anti-redundancia, dual-stack
  pandas/polars, ABI numpy/torch/sklearn). Use when adding or removing pip
  packages, editing requirements*.txt, or resolving dependency conflicts.
---

# Python deps

## Ordem

1. Ler `docs/engineering-python-deps.md`
2. Grep `import`/`from` do pacote no first-party (`app/`, `infra/docker/`)
3. Se so-transitivo de wrapper (ex. `coverage` via `pytest-cov`): nao declarar
4. Se import direto: declarar no requirements do ambiente certo
5. DataFrame: so `pandas` (meta) ou `polars` (DL); nunca ambos no mesmo modulo; nunca 3a lib DF
6. Atualizar `app/requirements.txt` e/ou `requirements-dev.txt` e/ou Docker reqs
7. WSL: `python -m pip check`
8. Se mexeu em numpy/torch/sklearn: smoke `import numpy,torch,sklearn,joblib,pandas,polars`
9. Rodar pre-commit / testes de politica `test_python_deps_policy`

## Anti-padroes

Pinar `coverage` junto de `pytest-cov`; remover `joblib` com imports vivos; migrar pandas/polars sem mandato; ignorar falha de ABI apos bump.
