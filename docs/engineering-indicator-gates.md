# Catalogo de gates (SSOT minimo)

Hot path vivo:

1. SKIP tecnico: `training` / `data` / `deploy` / `predict_error` / stop-win `EXEC_PAUSE` / cooldown pos-LOSS (ladder LIN) / pausa de sessao (3 LOSS em janela 5, 2 ciclos M5)
2. TCN decide CALL/PUT
3. Anti-loss loss-clf = FLIP por `p_eff` apos auto_learn (young 0.55 / mature 0.58; ancora TCN)
4. SCALE **retract/explos/tape adapt** (nao SKIP; ultima palavra): retract/explos com mi+mili alinhados; ou fita **forte** (`tape_strong`) → fixa EXEC (pode desfazer FLIP); discordance so telemetria
5. Kelly + SIDE_EQ sizing (META edge ≤ 0 → soft Kelly via `soft_veto_score_factor`; sem SKIP)

**Proibido:** quality gate / signal_skip / Soft Kelly do loss-clf / HARD SKIP / fusao EV / `cal_soft_edge`. IND RSI/ADX/HURST = telemetria.

Telemetria: uma linha `[GATES] || LOSS_CLF: FLIP|OK|off …` (`p=` cru, `pe=` efetivo se diferir, `floor=` piso de decisao). Pos-settle, FLIPs logam `LOSS_CLF || QUALITY flip=1 young= hit= p= pe= n=` (hit-rate do lado invertido; nao muda lado).

Protocolo: nao reabrir quality gate amplo; gate novo so com mandato explicito e SSOT nomeado.

Refs: playbook, `engineering-settings-ssot.md`, rule `aether-execution-gates.mdc`, skill `aether-binary-senior`.

## Inventario de indicadores (feature, nao gate)

`precompute_price_series` calcula series causais. O ganho operacional e **entrar no vetor**, nao virar SKIP. Catalogo de gates permanece fechado.

### TCN 14D (`dl_feature_orthogonal`)

`norm_log_ret_1`, `norm_log_ret_5`, `rsi_centered`, `delta_rsi`, `bb_pct_b_centered`, `bb_log_width`, `norm_atr`, `ema_dist_9_21`, `ema_dist_20_50`, `macd_hist_norm`, `stoch_k_centered`, `adx_scaled`, `realized_vol_ratio`, **`hurst_centered`** (substitui `macro_trend_ctx` estatico). Dimensao travada em 14; bump exige retreino + fail-closed.

### Loss-classifier 24D

Vetor causal in-place: Cal/raw, `cal_raw_discord`, regime SCALE (telemetria), meta `predicted_payoff_edge`, Hurst, ADX, `variance_ratio`, `keltner_deviation`, ATR/tick accel, `bb_width_z`. **Sem** bits circulares de lado EXEC/TCN, **sem** `scale_adapted`, **sem** `live_wr` de sessao. Schema 24D inalterado; `make docker-rebuild` invalida buffer/pkl apos troca de significado. Sidecar valida `schema_hash` + vetor finito; `/v1/learn` ignora `contract_id` repetido; `degenerate` (colapso/ECE extremo) so derruba `veto_ready` — **nao** vira SKIP/HARD SKIP.

### Meta 23D

14 TCN + 4 micro-vol z + 3 microestrutura `1HZ75V` (`micro_price_velocity`, `micro_tick_count_norm`, `implied_vol_centered`) + 2 flow (`micro_tick_acceleration`, `keltner_deviation_ratio`). **Nao** decide lado. Predict aplica `label_scale` do bundle (se houver) e clamp **[-1, +0.85]**. Teacher live = `profit/stake` do settle; `meta_online_*` nao substitui `meta_lgbm.pkl` so por `n_train`.

### Ocioso (calculado, nao no 14D)

`cci`, `williams_r`, `roc`, `roc_rsi`, `price_zscore`, `vol_z`, `stoch_d`, `di_diff`, `cmo`, `ema_dist_20`, `vol_vs_target`, `keltner_pct_b` (loss/meta leem Keltner via flow/`indicators.keltner`). Candidatos a **substituicao** no 14D so com evidencia OOS — nunca gate novo.

SCALE vision: telemetria + retract/explos adapt last apos FLIP (`adapt_retract_enabled`); IND RSI/ADX… so print.
