---
name: aether-dl-train
description: >-
  Diagnostica treino/inferencia DL do Aether (checkpoint lookback/granularity,
  ACC, label_mode, Triton sync, train_meta). Use when training fails, ACC gate
  trips, TorchScript/MinIO issues, or the user mentions train.py, lookback, or meta LightGBM.
---

# DL train / inferencia

## Ordem de diagnostico

1. Settings: lookback, micro 120s, `label_mode=ma_trend`, `deploy_gate.enabled` / `force_ok`
2. Checkpoint: feat_dim=34, lookback, `val_accuracy`, `deploy_ok`
3. Early stop: `min_epochs` / patience; restore **best val_acc** (nao so loss)
4. ACC: soft_min **0.53** — mini-deploy nao bypassa
5. `launch-train.bat`: apos DL roda `check_dl_deploy_gate.py` antes do meta
6. Meta: variance nula → Timescale flat; usar `--source auto` / Deriv; alvo payoff fallback
7. Triton/meta HTTP opcionais — confirmar flags

## Anti-padroes

Trocar label sem retreino; `force_ok=true`; treinar meta em hydrate sintetico; ignorar ACC; restaurar checkpoint so por loss.

Doc: `docs/engineering-deep-learning.md`
