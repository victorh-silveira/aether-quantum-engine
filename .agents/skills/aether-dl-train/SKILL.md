---
name: aether-dl-train
description: Treinamento e avaliacao da stack de pesquisa experimental de ML (TCN) em ambiente isolado.
---

# Pesquisa Experimental de Deep Learning

## 1. Contexto e Pré-requisitos
- Leia `docs/engineering-deep-learning.md` e `.agents/rules/aether-deep-learning.mdc`.
- **Importante:** Esta rotina pertence exclusivamente a pesquisa experimental (`make app-research-ml`) e nao afeta o caminho operacional de contratos Rise/Fall M15.

## 2. Procedimento Operacional Passo a Passo
1. **Preparação de Features e Datasets:**
   - Assegure que as matrizes de features sejam construidas com Polars e normalizadas sem vazamento de dados futuros.
2. **Treinamento e Validação Walk-Forward:**
   - Execute o pipeline de treino walk-forward com periodos de embargo estritos entre os folds.
3. **Métricas de Calibração:**
   - Avalie ECE (Expected Calibration Error) e Brier Score dos modelos. Modelos com val_acc proxima de 0,5 sao automaticamente reprovados.
4. **Exportação de Checkpoint:**
   - Salve checkpoints aprovados contendo todos os metadados de treino em `data/dl/` ou MinIO.

## 3. Comandos de Verificação
```bash
/mnt/c/Users/victor-silveira/anaconda3/envs/deriv-api/python.exe scripts/operations/check_dl_checkpoint.py
/mnt/c/Users/victor-silveira/anaconda3/envs/deriv-api/python.exe -m pytest app/tests/unit/application/test_deep_learning*.py -v
```

## 4. Critérios de Aceite
- Treinamento finalizado sem vazamento de dados temporais.
- Zero impacto no runtime operacional do motor.
