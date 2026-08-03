---
name: aether-ops-runbook
description: >-
  Runbook operacional Aether (launch-all, train, clean_workspace, monitores
  redis/state). Use when starting the engine, training batch, cleaning workspace,
  or running monitor scripts under app/scripts.
---

# Ops runbook

## Entradas

| Objetivo | Onde |
|----------|------|
| Rodar motor | `run.py` / scripts `batch/launch-*` |
| Treino DL | `train.py` / `batch/launch-train*` |
| Meta offline | `scripts/operations/train_meta_*` |
| Pre-commit motor | `scripts/operations/clean_workspace.py` |
| Monitor | `scripts/monitor/*` |

## Regras

- Executar no **WSL**
- Nao versionar caches limpos pelo stage clean
- Antes de launch live: Docker core saudavel + settings revisados

Docs: `docs/structure.md` §Scripts, `docs/engineering-standards.md`
