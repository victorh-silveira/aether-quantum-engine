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
| Treino DL | `train.py` / `batch/launch-train*` (etapa 0: `sanitize_fresh_run`) |
| Meta offline | `scripts/operations/train_meta_*` |
| Sanitizar run | `scripts/operations/sanitize_fresh_run.py` / `make sanitize-run` |
| Pre-commit motor | `scripts/operations/clean_workspace.py` |
| Monitor | `scripts/monitor/*` |

## Regras

- Executar no **WSL**
- Nao versionar caches limpos pelo stage clean
- Antes de launch live: Docker core saudavel + settings revisados
- Ciclo fresco: `make docker-reset` → `launch-train.bat` → `launch-all-*` (train limpa checkpoints; reset limpa volumes)
- Treino rejeitado pelo gate **nao** preserva checkpoint anterior; `deploy_ok=false` aborta meta

Docs: `docs/structure.md` §Scripts, `docs/engineering-standards.md`
