---
name: aether-dl-train
description: >-
  Diagnostica treino/inferencia DL do Aether (checkpoint lookback/granularity,
  ACC, label_mode, train_meta). Use when training fails, ACC gate
  trips, TorchScript/MinIO issues, or the user mentions train.py, lookback, or meta LightGBM.
---

# DL train / inferencia

## Ordem de diagnostico

1. Settings: lookback **30**, macro **86400s** (D1 / 365 barras), micro **300s** (M5, `training_history_bars` **2000**), label $N=1$ vela M5 (`label_horizon_bars=1`), contrato ops **5 m** (`params.duration=5`), `label_mode=quantum_multi_barrier`, `deploy_gate.enabled` / `force_ok` / `allow_undeployed_inference` **false**
2. Telemetria de lado: `label_call_frac` / `pred_call_frac` / `minority_recall` no treino
3. Balance: `deep_learning.sample_weighting.class_balance_*` via `compose_train_weights`
4. Recency: `recency_enabled` / `recency_half_life_n` (default **500**)
5. Deploy collapse: `reject_majority_collapse` — pred skew (`|pred-0.5|` / `|pred-label|` > **0.20**) rejeita sozinho; label skew + `min_minority_recall` (**0.25**)
6. Checkpoint: feat_dim=**14** (ultima dim `hurst_centered`), lookback **30**, granularity micro **300** (treino lean M5; macro D1 **86400** no buffer), `val_accuracy`, `deploy_ok`; ckpt 34D / meta 43D = invalidos
7. Early stop: `min_epochs` **15** / patience **17**; restore pico de validação; sharp sem colapso
8. ACC: soft_min **0.55** no path label; deploy_gate fail-closed
9. Brier: `max_brier` **0.28** (= `soft_max_brier`); sharpness `min_oos_sharpness` **0.03**
10. Fail-closed: export falhou → `train.py` exit!=0; gate rejeita ckpt com lookback/granularity != settings; meta nao roda
11. `launch-train.bat`: apos DL roda `check_dl_deploy_gate.py`; depois `ensure_timescale` seed Deriv (**M5×5000 + D1×365**, timeout **900s**, persist lote) antes do meta. Lean startup treino micro pede `max(fetch_count, micro_fetch_count, training_history_bars)` (**2000**, nao 500).
12. Meta: LightGBM **23D** (microestrutura `1HZ75V`, sem peer); `--bars` **5000** (micro M5; nao confundir com 365 D1); Timescale smoke → Deriv INFO; teacher = payoff assinado **cru ate o split**; unit-scale O(1) = std do **fold de treino** (`label_scale` / `train_std` / `val_std`; nao pontos brutos, sem escala no y completo). Prefixo `|payoff|~0` ou split `val_std/train_std > 1.5` ou mediana L1 `val_mae/train_mae > 2.0` e cortado; se ainda degenerado, falha fechado. Objetivo **regression_l1** (mesmo MAE do gate); n grande: **80** rounds, early-stop **L1** patience **15**, depth **1**, `bagging_fraction`. Export usa o snapshot com `val_mae/train_mae<=2.0` de menor val L1 (snapshot 0 = mediana L1 do treino; z/IR no mesmo snapshot; val purged completa, sem stride 5). Export bloqueado se todos os trials furam o teto (mensagem com `label_mode` / `y_std` / `train_std` / `val_std` / `med_mae_gap` / `med_naive_mae`); **nao** baixar o teto 2.0 nem inflar `--bars` alem de 5000.
13. Meta HTTP opcional — confirmar flags; TCN = eager/CUDA local no host (`inference_mode`; nao bloquear o event loop)
14. Universo runtime = **1HZ75V**; contrato ops fixo **5 m**
15. Run fresca: `sanitize_fresh_run` no inicio de `launch-train`; `make docker-reset` sanitiza + volumes
16. Anti-overfit: `weight_decay` **0.005**, `tcn.dropout` **0.20**, `label_smoothing` **0.02**, `learning_rate` **0.001**
17. Pos-treino: `make docker-rebuild` recarrega meta/loss **sem** apagar `data/dl`
18. Cal overconfident: `apply_calibrator_stable` devolve raw se margem raw > Cal; clipa p_call em `[raw±max_calibrated_raw_gap]` (**0.05**) **antes** do ramo `raw_extreme`; `min_calibration_margin_floor` **0.05**; TCN live sempre CALL se Cal ≥**0.5** senao PUT (sem `SKIP:NEUTRAL_ZONE`); call/put **0.55/0.45** nao skipam; banda `[0.45, 0.55]` so telemetria/`raw_extreme`; flag `cal_raw_gap_capped`; `temperature_min` **0.75**
19. Optuna/tuning offline — nao disputar VRAM com inferencia live; artefatos no MinIO
20. Pos-launch: telemetria `label_call_frac` / `pred_call_frac` / `minority_recall`; vies de classe corrige-se com sample_weighting + majority-collapse — **nao** com flip live

## Anti-padroes

Trocar label sem retreino; `force_ok=true`; treinar meta em hydrate sintetico; ignorar ACC no path label; restaurar checkpoint so por loss; tratar vies de classe com veto de sinal live em vez de balance/recency/collapse; baixar `min_oos_sharpness` para “passar” export; baixar `META_EXPORT_MAX_MAE_GAP` (2.0) para passar Optuna; tratar `[SUCESSO]` do bat se o gate/treino falhou; operar checkpoints descalibrados.

Com `online_training=false` (SSOT): DEMO sobe com checkpoint do `launch-train` e nao retreina TCN em runtime. Loss/meta `/learn` a cada trade.

Doc: `docs/engineering-deep-learning.md` + `docs/engineering-architecture-senior.md`
