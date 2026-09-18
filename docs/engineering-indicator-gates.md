# Catalogo de gates (SSOT minimo)

Hot path vivo:

1. SKIP tecnico: `training` / `data` / `deploy` / `predict_error` / stop-win `EXEC_PAUSE` / cooldown pos-LOSS (ladder LIN) / pausa de sessao (3 LOSS em janela 5, 2 ciclos M5)
2. SKIP sinal: `neg_edge` se Cal Edge ≤ 0 (`skip_neg_edge` **true**); waived com PEND ≥ `material_pending_min` ou smart waive (Edge >= -0.030 com loss_clf p_loss <= 0.485 ou loss-clf sob bootstrap com margem direcional ativa); `acc_floor` so quando `skip_below_soft_min_acc` **true** (ops **false**)
3. TCN decide CALL/PUT (Cal ≥ 0.5 → CALL)
4. Anti-loss loss-clf = FLIP por `p_eff` apos auto_learn e `n_train >= 12` (young/mature 0.58; ancora TCN); **sem** bloqueio por vela
5. SCALE adapt **off** (`adapt_retract_enabled` **false**); `skip_doji` / `skip_exec_vs_candle` / `skip_scale_candle_discord` **false**
6. `invert_exec_side` **false**
7. Kelly + SIDE_EQ sizing (META **nao** soft Kelly)

**Proibido:** quality gate amplo / Soft Kelly do loss-clf ou META / HARD SKIP loss-clf / fusao EV / `cal_soft_edge` / SCALE adapt de lado. IND RSI/ADX/HURST = fora do log de ciclo.

Telemetria ciclo: CLUSTER → `[GATES] || LOSS_CLF` → KELLY → EXEC → RESOLVED.

Protocolo: gate novo so com mandato explicito e SSOT nomeado.

Refs: playbook, `engineering-settings-ssot.md`, rule `aether-execution-gates.mdc`, skill `aether-binary-senior`.

## Inventario de indicadores (feature, nao gate)

`precompute_price_series` calcula series causais. O ganho operacional e **entrar no vetor**, nao virar SKIP. Catalogo de gates permanece fechado.

### TCN 14D (`dl_feature_orthogonal`)

`norm_log_ret_1`, `norm_log_ret_5`, `rsi_centered`, `delta_rsi`, `bb_pct_b_centered`, `bb_log_width`, `norm_atr`, `ema_dist_9_21`, `ema_dist_20_50`, `macd_hist_norm`, `stoch_k_centered`, `adx_scaled`, `realized_vol_ratio`, **`hurst_centered`**. Dimensao travada em 14; bump exige retreino + fail-closed.

### Loss-classifier 24D

Vetor causal in-place: Cal/raw, `cal_raw_discord`, regime SCALE (telemetria), meta `predicted_payoff_edge`, Hurst, ADX, `variance_ratio`, `keltner_deviation`, ATR/tick accel, `bb_width_z`. Schema 24D inalterado.

### Meta 23D

14 TCN + microestrutura. **Nao** decide lado nem comprime stake live.

### Ocioso (calculado, nao no 14D)

`cci`, `williams_r`, `roc`, etc. Candidatos a substituicao no 14D so com evidencia OOS — nunca gate novo.
