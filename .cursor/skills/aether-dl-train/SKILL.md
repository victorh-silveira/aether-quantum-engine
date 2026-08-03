---
name: aether-dl-train
description: >-
  Diagnostica treino/inferencia DL do Aether (checkpoint lookback/granularity,
  ACC, label_mode, Triton sync, train_meta). Use when training fails, ACC gate
  trips, TorchScript/MinIO issues, or the user mentions train.py, lookback, or meta LightGBM.
---

# DL train / inferencia

## Ordem de diagnostico

1. Settings: lookback, granularity macro/micro, label_mode
2. Checkpoint: feat_dim=34, lookback e granularity compativeis (`dl_startup`)
3. Device CUDA vs CPU; logs `DL:`
4. ACC / val_accuracy vs `min_validation_accuracy_gate`
5. Triton/meta opcionais — confirmar `enabled` / require flags
6. Meta offline: `app/scripts/operations/train_meta_*`

## Anti-padroes

Trocar label sem retreino; ignorar ACC; forcar trade por ACC baixo.

Doc: `docs/engineering-deep-learning.md`
