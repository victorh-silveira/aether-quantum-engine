---
name: aether-ops-runbook
description: Runbook operacional Aether — execucao, treinamento, sanitizacao e rotinas sob WSL.
---

# Runbook de Operações do Sistema

## 1. Contexto e Pré-requisitos
- Leia `AGENTS.md` e `docs/structure.md`.

## 2. Procedimento Operacional Passo a Passo
1. **Inicialização do Sistema Completo:**
   ```bash
   make docker-up
   make app-run
   ```
2. **Execução de Diagnóstico / Treinamento:**
   ```bash
   make app-train
   ```
3. **Sanitização de Ambiente:**
   ```bash
   make app-clean
   ```

## 3. Comandos de Verificação
```bash
/mnt/c/Users/victor-silveira/anaconda3/envs/deriv-api/python.exe app/scripts/operations/check_dl_checkpoint.py
```

## 4. Critérios de Aceite
- Operacoes executadas via comandos oficiais sem travamentos.
