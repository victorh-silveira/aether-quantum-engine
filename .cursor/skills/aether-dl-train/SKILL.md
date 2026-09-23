---
name: aether-dl-train
description: >-
  Diagnostica treino/inferencia DL do Aether (checkpoint lookback/granularity,
  ACC, label_mode, train_meta). Use when training fails, ACC gate
  trips, TorchScript/MinIO issues, or the user mentions train.py, lookback, or meta LightGBM.
---

# DL train / inferencia

## Ordem de diagnostico

1. Settings: lookback **30**, macro **86400s** (D1 / 365 barras), micro **300s** (M5, `training_history_bars` **5000**), label $N=1$ vela M5 (`label_horizon_bars=1`), contrato ops **5 m** (`params.duration=5`), `label_mode=quantum_multi_barrier`; `force_ok` exporta so para diagnostico, promocao exige ACC anti-colapso >=**0.50**, Brier settlement **<0.245**, N>=**48**, LCB Wilson unilateral 90% >= breakeven e anti-collapse
2. Telemetria de lado: `label_call_frac` / `pred_call_frac` / `minority_recall` no treino
3. Balance: `deep_learning.sample_weighting.class_balance_*` via `compose_train_weights`
4. Recency: `recency_enabled` / `recency_half_life_n` (default **500**)
5. Deploy collapse: `reject_majority_collapse` **true**; pred skew >**0.15** ou `minority_recall` <**0.40** rejeita promocao
6. Checkpoint: feat_dim=**14** (ultima dim `hurst_centered`), lookback **30**, granularity micro **300** (treino lean M5; macro D1 **86400** no buffer), `val_accuracy`, `deploy_ok`; ckpt 34D / meta 43D = invalidos
7. Early stop: `min_epochs` **15** / patience **12**; restore **maior val_acc**; sharp so se ACC sharp >= pico e sem `collapse_hit`
8. Deploy treino: `force_ok=true` conserva artefato diagnostico; `resolve_deploy_ok` exige qualidade temporal e o gate bloqueia meta quando a promocao reprova
9. Brier: settlement Brier **<0.245**; o checkpoint precisa persistir `deploy_settlement_wilson_lcb` acima do breakeven configurado
10. Provisorio: se N>=**48**, WR settlement>=**0.57** e Brier<**0.245**, mas o LCB Wilson pleno falhar, `deploy_provisional` permite somente stake <=**1%** da banca; nao altera o gate pleno
10. Fail-closed so em geometria / I/O; export fraco gera WARNING e segue; meta/loss bootstrap sempre gravam
11. `launch-train.bat`: apos DL roda `check_dl_deploy_gate.py` (`force_ok`); depois `ensure_timescale` seed Deriv (**M5×5000 + D1×365**) antes do meta. Lean startup treino micro pede `max(fetch_count, micro_fetch_count, training_history_bars)` (**2000**).
12. Meta: LightGBM **23D**; `--bars` **5000**; teacher payoff unit-scale no fold de treino; export exige z-score **0.01** e IR **0.05** e gap MAE <=**1.50**. Loss bootstrap sempre escreve pkl.
13. Meta HTTP opcional — confirmar flags; TCN = eager/CUDA local no host (`inference_mode`; nao bloquear o event loop)
14. Universo runtime = **1HZ75V**; contrato ops fixo **5 m**
15. Run fresca: `sanitize_fresh_run` no inicio de `launch-train`; `make docker-reset` sanitiza + volumes
16. Anti-overfit: `weight_decay` **0.01**, `tcn.dropout` **0.25**, `label_smoothing` **0.0**, `learning_rate` **0.001**, `focal_gamma` **1.0**, `lr_scheduler` **reduce_on_plateau**, `aux_regression_weight` **0.08**
17. Pos-treino: `make docker-rebuild` recarrega meta/loss **sem** apagar `data/dl`
18. Cal overconfident: `apply_calibrator_stable` devolve raw se margem raw > Cal ou Cal < floor **0.05** com raw nitido; ambos moles → mais nitido sem stretch; clipa p_call em `[raw±max_calibrated_raw_gap]` (**0.05**) **antes** do ramo `raw_extreme`; TCN live sempre CALL se Cal ≥**0.5** senao PUT; call/put **0.55/0.45** nao skipam; banda `[0.45, 0.55]` so telemetria/`raw_extreme`; `temperature_min` **0.75**
19. Optuna/tuning offline — nao disputar VRAM com inferencia live; artefatos no MinIO
20. Pos-launch: telemetria `label_call_frac` / `pred_call_frac` / `minority_recall`
21. Edge≤0 com Cal~0.52 e `skip_neg_edge`: processo correto; limpar `EXEC_EMPTY` exige **retreino + export** (sempre no launch-train) e novo TorchScript no MinIO

## Anti-padroes

Trocar label sem retreino; desligar `force_ok` sem mandato; treinar meta em hydrate sintetico; Soft Kelly / desligar `skip_neg_edge` para “passar” Edge mole; tratar `[SUCESSO]` do bat se geometria/I/O falhou.

Com `online_training=false` (SSOT): DEMO sobe com checkpoint do `launch-train` e nao retreina TCN em runtime. Loss/meta `/learn` a cada trade.

Doc: `docs/engineering-deep-learning.md` + `docs/engineering-architecture-senior.md`
