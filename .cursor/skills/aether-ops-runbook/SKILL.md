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
| Treino DL | `train.py` / `batch/launch-train*` (sanitize → sweep H15–H60 + promote → gate → Timescale → meta; logs densos / celula CRITICAL) |
| Meta offline | `scripts/operations/train_meta_*` |
| Sanitizar run | `scripts/operations/sanitize_fresh_run.py` / `make sanitize-run` |
| Pre-commit motor | `scripts/operations/clean_workspace.py` |
| Monitor | `scripts/monitor/*` |

## Regras

- Executar no **WSL**
- Nao versionar caches limpos pelo stage clean
- Antes de launch live: Docker core saudavel + settings revisados
- Diario: `make docker-up` → `launch-train.bat` → `make docker-rebuild` → `launch-all-*` (rebuild recarrega pkls; **nao** apaga TCN)
- Ciclo fresco: `make docker-reset` → `launch-train.bat` → `make docker-rebuild` → `launch-all-*` (reset limpa volumes/TCN; train gera checkpoints)
- Treino rejeitado pelo gate **nao** preserva checkpoint anterior; `deploy_ok=false` aborta meta

Docs: `docs/structure.md` §Scripts, `docs/engineering-standards.md`
