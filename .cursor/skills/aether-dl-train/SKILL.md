---
name: aether-dl-train
description: >-
  Diagnostica treino/inferencia DL do Aether (checkpoint lookback/granularity,
  ACC, label_mode, Triton sync, train_meta). Use when training fails, ACC gate
  trips, TorchScript/MinIO issues, or the user mentions train.py, lookback, or meta LightGBM.
---

# DL train / inferencia

## Ordem de diagnostico

1. Settings: lookback **720**, micro **120s** (M2), contrato **2 m**, `label_mode=ma_trend`, `deploy_gate.enabled` / `force_ok`
2. Telemetria de lado: `label_call_frac` / `pred_call_frac` / `minority_recall` no treino
3. Balance: `deep_learning.sample_weighting.class_balance_*` via `compose_train_weights`
4. Recency: `recency_enabled` / `recency_half_life_n` (default **2000**)
5. Deploy collapse: `reject_majority_collapse` + `max_label_call_frac_bias` (0.20) + `min_minority_recall` (0.25)
6. Checkpoint: feat_dim=34, lookback **720**, granularity micro **120**, `val_accuracy`, `deploy_ok`
7. Early stop: `min_epochs` / patience; restore **best val_acc** (nao so loss)
8. ACC: soft_min **0.53** — mini-deploy nao bypassa
9. Sharpness: `min_oos_sharpness` **0.01**; se temperature/Platt colapsar → fit cai para `identity` (nao baixar o piso)
10. Fail-closed: export falhou → `train.py` exit!=0; gate rejeita ckpt com lookback/granularity != settings; meta nao roda
11. `launch-train.bat`: apos DL roda `check_dl_deploy_gate.py` antes do meta
12. Meta: variance nula → Timescale flat; usar `--source auto` / Deriv; alvo payoff fallback; gran meta **120s**
13. Triton/meta HTTP opcionais — confirmar flags
14. Pos-migrate M2: invalidar pth/TorchScript antigos (gran 60/300/900 ou contrato 15m), re-hydrate Timescale **120/3600**, retreinar TCN+meta
15. Universo **`R_10`**: artefactos Volatility/legado invalidos; sync MinIO/Triton com nome `R_10`
16. Gap TCN→meta: `launch-train` usa `ensure_timescale.py --check-only` (sem seed Deriv entre etapas); bootstrap wait cap **30 s**; shortfall API ≥ **95%** do alvo (`train_history_shortfall_ratio`) em TCN **e** meta (ex.: 1984/2000)


## Anti-padroes

Trocar label sem retreino; `force_ok=true`; treinar meta em hydrate sintetico; ignorar ACC; restaurar checkpoint so por loss; tratar vies de classe com veto de sinal live em vez de balance/recency/collapse; baixar `min_oos_sharpness` para “passar” export; tratar `[SUCESSO]` do bat se o gate/treino falhou; operar Volatility checkpoints no simbolo `R_10`.

Com `online_training=true` (SSOT): DEMO sobe com checkpoint se existir; retreino TCN deferido a cada settle + vela/rolling. Loss/meta `/learn` a cada trade.

Doc: `docs/engineering-deep-learning.md`
