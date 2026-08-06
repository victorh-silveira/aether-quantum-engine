---
name: aether-dl-train
description: >-
  Diagnostica treino/inferencia DL do Aether (checkpoint lookback/granularity,
  ACC, label_mode, Triton sync, train_meta). Use when training fails, ACC gate
  trips, TorchScript/MinIO issues, or the user mentions train.py, lookback, or meta LightGBM.
---

# DL train / inferencia

## Ordem de diagnostico

1. Settings: lookback **720**, micro **900s** (M15), contrato **15 m**, `label_mode=ma_trend`, `deploy_gate.enabled` / `force_ok`
2. Telemetria de lado: `label_call_frac` / `pred_call_frac` / `minority_recall` no treino
3. Balance: `deep_learning.sample_weighting.class_balance_*` via `compose_train_weights`
4. Recency: `recency_enabled` / `recency_half_life_n` (default **2000**)
5. Deploy collapse: `reject_majority_collapse` + `max_label_call_frac_bias` (0.20) + `min_minority_recall` (0.25)
6. Checkpoint: feat_dim=34, lookback **720**, granularity micro **900**, `val_accuracy`, `deploy_ok`
7. Early stop: `min_epochs` / patience; restore **best val_acc** (nao so loss)
8. ACC: soft_min **0.53** — mini-deploy nao bypassa
9. Sharpness: `min_oos_sharpness` **0.01**; se temperature/Platt colapsar → fit cai para `identity` (nao baixar o piso)
10. Fail-closed: export falhou → `train.py` exit!=0; gate rejeita ckpt com lookback/granularity != settings; meta nao roda
11. `launch-train.bat`: apos DL roda `check_dl_deploy_gate.py` antes do meta
12. Meta: variance nula → Timescale flat; usar `--source auto` / Deriv; alvo payoff fallback; gran meta **900s**
13. Triton/meta HTTP opcionais — confirmar flags
14. Pos-migrate M15: invalidar pth/TorchScript antigos (gran 60/300 ou contrato 30s), re-hydrate Timescale **900/3600**, retreinar TCN+meta
15. Universo **`OTC_SPC`**: artefactos Volatility (`R_10`) invalidos; sync MinIO/Triton com nome `OTC_SPC`
16. Gap TCN→meta: `launch-train` usa `ensure_timescale.py --check-only` (sem seed Deriv entre etapas); bootstrap wait cap **30 s** (`bootstrap_history_wait_cap_seconds`)

## Anti-padroes

Trocar label sem retreino; `force_ok=true`; treinar meta em hydrate sintetico; ignorar ACC; restaurar checkpoint so por loss; tratar vies de classe com veto de sinal live em vez de balance/recency/collapse; baixar `min_oos_sharpness` para “passar” export; tratar `[SUCESSO]` do bat se o gate/treino falhou; operar Volatility checkpoints no simbolo `OTC_SPC`.

Com `online_training=false` (SSOT atual): o motor **nao** treina no DEMO/`launch-all-demo` — rode `launch-train.bat` (TCN+meta) antes. Checkpoint incompativel → SKIP tecnico ate treino offline.

Doc: `docs/engineering-deep-learning.md`
