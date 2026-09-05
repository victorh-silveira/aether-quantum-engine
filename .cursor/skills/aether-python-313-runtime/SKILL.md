---
name: aether-python-313-runtime
description: >-
  Audita runtime CPython 3.13 Aether (GIL, GC, specializing interpreter,
  typing, buffers, profiling). Use when changing interpreter build, hot-path
  allocation, free-threaded experiments, or reviewing Python runtime PRs.
---

# Runtime CPython 3.13

## Quando aplicar

Mudanca de interpretador/Conda, churn de memoria no ciclo, free-threaded, tipagem de ports, profiling do motor, ou review de PR que toque hot path Python puro.

## Checklist

1. Ler `docs/engineering-python-313-runtime.md` + `docs/engineering-architecture-senior.md`
2. Confirmar CPython **3.13.x com GIL** no WSL/Conda `deriv-api` (nao free-threaded em prod)
3. Hot path: minimizar alocacoes pequenas (PyMalloc/arenas) e grafos ciclicos (GC STW Gen0-2)
4. Handlers quentes monomorficos (Tier 1/2 specializing) — evitar despacho polimorfico por tick
5. Typing: Protocols nas ports; ParamSpec em wrappers; type params 3.12+; sem metaclasse pesada
6. Buffers: `memoryview` / PEP 688; zero-copy na borda quando seguro
7. Profile com **py-spy** ou **austin** + `tracemalloc`; nao cProfile no live
8. Domain puro e Polars-only respeitados (skills irmas)
9. Fechar superficie com `aether-surface-sync` se docs/rules mudaram

## Anti-padroes

- Free-threaded (`--disable-gil`) em DEMO/prod
- cProfile acoplado ao event loop live
- Metaclasses onde `__init_subclass__`/descriptors bastam
- Ignorar pausas GC sob churn de dicts/DataFrames no ciclo
- Polimorfismo agressivo em handlers WS/gates/settle

## Refs no repo

- `docs/engineering-python-313-runtime.md`
- `docs/engineering-architecture-senior.md`
- `docs/engineering-python-deps.md`
- `AGENTS.md`
- `.cursor/rules/aether-python-313-runtime.mdc`
- `.cursor/rules/aether-architecture-senior.mdc`

## Skills irmas

`aether-asyncio-supervisor`, `aether-polars-arrow`, `aether-torch-cuda-infer`, `aether-asyncpg-timescale`, `aether-redis-hiredis`, `aether-python-deps`, `aether-architecture-senior`, `aether-surface-sync`
