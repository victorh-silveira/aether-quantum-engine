---
name: aether-dl-train
description: >-
  Diagnostica treino/inferencia DL do Aether (checkpoint lookback/granularity,
  ACC, label_mode, train_meta). Use when training fails, ACC gate
  trips, TorchScript/MinIO issues, or the user mentions train.py, lookback, or meta LightGBM.
---

# DL train / inferencia

## Ordem de diagnostico

1. Settings: lookback **30**, macro **86400s** (D1 / 365 barras), micro **300s** (M5), label $N=1$ vela M5 (`label_horizon_bars=1`), contrato ops **5 m** (`params.duration=5`), `label_mode=quantum_multi_barrier`, `deploy_gate.enabled` / `force_ok`
2. Telemetria de lado: `label_call_frac` / `pred_call_frac` / `minority_recall` no treino
3. Balance: `deep_learning.sample_weighting.class_balance_*` via `compose_train_weights`
4. Recency: `recency_enabled` / `recency_half_life_n` (default **500**)
5. Deploy collapse: `reject_majority_collapse` — pred skew (`|pred-0.5|` / `|pred-label|` > **0.25**) rejeita sozinho; label skew + `min_minority_recall` (**0.25**)
6. Checkpoint: feat_dim=34, lookback **30**, granularity macro **86400**, `val_accuracy`, `deploy_ok`
7. Early stop: `min_epochs` **15** / patience **17**; restore pico de validação; sharp sem colapso
8. ACC: soft_min **0.55** no path label; deploy_gate fail-closed
9. Brier: `max_brier` **0.28** (= `soft_max_brier`); sharpness `min_oos_sharpness` **0.01**
10. Fail-closed: export falhou → `train.py` exit!=0; gate rejeita ckpt com lookback/granularity != settings; meta nao roda
11. `launch-train.bat`: apos DL roda `check_dl_deploy_gate.py` antes do meta
12. Meta: LightGBM **43D**; `--bars` **5000** (micro M5; nao confundir com 365 D1); variance nula → Timescale flat; `--source auto` / Deriv; alvo payoff fallback
13. Meta HTTP opcional — confirmar flags; TCN = eager/CUDA local no host (`inference_mode`; nao bloquear o event loop)
14. Universo runtime = **1HZ75V**; contrato ops fixo **5 m**
15. Run fresca: `sanitize_fresh_run` no inicio de `launch-train`; `make docker-reset` sanitiza + volumes
16. Anti-overfit: `weight_decay` **0.005**, `tcn.dropout` **0.35**, `learning_rate` **0.001**
17. Pos-treino: `make docker-rebuild` recarrega meta/loss **sem** apagar `data/dl`
18. Cal overconfident: live clipa p_call em `[raw±max_calibrated_raw_gap]` (**0.08**); flag `cal_raw_gap_capped`; `temperature_min` **1.0**
19. Optuna/tuning offline — nao disputar VRAM com inferencia live; artefatos no MinIO

## Anti-padroes

Trocar label sem retreino; `force_ok=true`; treinar meta em hydrate sintetico; ignorar ACC no path label; restaurar checkpoint so por loss; tratar vies de classe com veto de sinal live em vez de balance/recency/collapse; baixar `min_oos_sharpness` para “passar” export; tratar `[SUCESSO]` do bat se o gate/treino falhou; operar checkpoints descalibrados.

Com `online_training=false` (SSOT): DEMO sobe com checkpoint do `launch-train` e nao retreina TCN em runtime. Loss/meta `/learn` a cada trade.

Doc: `docs/engineering-deep-learning.md` + `docs/engineering-architecture-senior.md`
