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
2. Telemetria de lado: `label_call_frac` / `pred_call_frac` / `minority_recall` no treino
3. Balance: `deep_learning.sample_weighting.class_balance_*` via `compose_train_weights`
4. Recency: `recency_enabled` / `recency_half_life_n` (default **2000**)
5. Deploy collapse: `reject_majority_collapse` + `max_label_call_frac_bias` (0.20) + `min_minority_recall` (0.25)
6. Checkpoint: feat_dim=34, lookback, `val_accuracy`, `deploy_ok`
7. Early stop: `min_epochs` / patience; restore **best val_acc** (nao so loss)
8. ACC: soft_min **0.53** — mini-deploy nao bypassa
9. `launch-train.bat`: apos DL roda `check_dl_deploy_gate.py` antes do meta
10. Meta: variance nula → Timescale flat; usar `--source auto` / Deriv; alvo payoff fallback
11. Triton/meta HTTP opcionais — confirmar flags

## Anti-padroes

Trocar label sem retreino; `force_ok=true`; treinar meta em hydrate sintetico; ignorar ACC; restaurar checkpoint so por loss; tratar vies de classe com veto de sinal live em vez de balance/recency/collapse.

Doc: `docs/engineering-deep-learning.md`
