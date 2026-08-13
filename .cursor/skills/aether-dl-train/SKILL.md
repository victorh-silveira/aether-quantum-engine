---
name: aether-dl-train
description: >-
  Diagnostica treino/inferencia DL do Aether (checkpoint lookback/granularity,
  ACC, label_mode, Triton sync, train_meta). Use when training fails, ACC gate
  trips, TorchScript/MinIO issues, or the user mentions train.py, lookback, or meta LightGBM.
---

# DL train / inferencia

## Ordem de diagnostico

1. Settings: lookback **480**, micro **180s** (M3), contrato **3 m**, `label_mode=ma_trend`, `deploy_gate.enabled` / `force_ok`
2. Telemetria de lado: `label_call_frac` / `pred_call_frac` / `minority_recall` no treino
3. Balance: `deep_learning.sample_weighting.class_balance_*` via `compose_train_weights`
4. Recency: `recency_enabled` / `recency_half_life_n` (default **2000**)
5. Deploy collapse: `reject_majority_collapse` — pred skew (`|pred-0.5|` / `|pred-label|` > **0.20**) rejeita sozinho; label skew + `min_minority_recall` (**0.25**)
6. Checkpoint: feat_dim=34, lookback **480**, granularity micro **180**, `val_accuracy`, `deploy_ok`
7. Early stop: `min_epochs` **20** / patience **16**; restore so pico com **BCE** val CE&lt;**0.70** (monitor de val **ignora** `focal_gamma`; nao epoca 1 com loss 0.80); sharp sem collapse
8. ACC: soft_min **0.53** no path label; gate pos-promote tambem aceita **settle_wr** elegivel (be+0.03, n≥16, history≥800)
9. Brier mini: `max_brier` **0.26** (= `soft_max_brier`); sharpness `min_oos_sharpness` **0.01**
10. Fail-closed: export falhou → `train.py` exit!=0; gate rejeita ckpt com lookback/granularity != settings; meta nao roda
11. `launch-train.bat`: apos DL roda `check_dl_deploy_gate.py` antes do meta
12. Meta: variance nula → Timescale flat; usar `--source auto` / Deriv; alvo payoff fallback; gran meta = micro SSOT (hoje **180s**)
13. Triton/meta HTTP opcionais — confirmar flags
14. Pos-promote TF: invalidar pth/TorchScript de gran anterior, re-hydrate Timescale no micro/macro novos, retreinar TCN+meta (ja feito no launch-train)
15. Universo **`R_10`**: artefactos Volatility/legado invalidos; sync MinIO/Triton com nome `R_10`
16. Gap TCN→meta: `launch-train` usa `ensure_timescale.py --check-only` (sem seed Deriv entre etapas); bootstrap wait cap **30 s**; shortfall API ≥ **95%** do alvo (`train_history_shortfall_ratio`) em TCN **e** meta
17. Run fresca: `sanitize_fresh_run` no inicio de `launch-train`; `make docker-reset` sanitiza + volumes; treino **nao** preserva checkpoint anterior
18. Anti-overfit R_10: `weight_decay` **0.001**, `tcn.dropout` **0.25**, `learning_rate` **0.001**, `recency_half_life_n` **2000**
19. Retries: `train_deploy_retries` **1** no sweep lean (reseed so se knob >1)
20. Pos-treino: `make docker-rebuild` recarrega meta/loss **sem** apagar `data/dl` (nao chamar `sanitize-run` depois do train)
21. Sweep multi-TF no **launch-train**: escala wall-clock (ancora M2); elegivel so **settle_wr** ≥ be+0.03 com **settle_n≥16** e **history≥800**; 1 tentativa/TF; promote carimba `deploy_ok`; gate pos-promote alinhado ao settle
22. Cal overconfident: live clipa p_call em `[raw±max_calibrated_raw_gap]` (**0.08**); flag `cal_raw_gap_capped`; `temperature_min` **1.0**; `tcn_pos_edge` exige raw_edge ≥ `fusion_min_edge_execute` (**0.04**) — sintoma CLUSTER Prob≈BE + p_lado≫0.7 + `why=tcn_pos_edge` = regressao

## Anti-padroes

Trocar label sem retreino; `force_ok=true`; treinar meta em hydrate sintetico; ignorar ACC no path label sem settle elegivel; restaurar checkpoint so por loss; tratar vies de classe com veto de sinal live em vez de balance/recency/collapse; baixar `min_oos_sharpness` para “passar” export; tratar `[SUCESSO]` do bat se o gate/treino falhou; operar Volatility checkpoints no simbolo `R_10`; promover TF com **settle_wr** abaixo de be+margem ou N/historico abaixo do piso; trocar TF ao vivo por ciclo.

Com `online_training=false` (SSOT): DEMO sobe com checkpoint do `launch-train` e nao retreina TCN em runtime. Loss/meta `/learn` a cada trade.

Doc: `docs/engineering-deep-learning.md`
