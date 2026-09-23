# Catalogo de gates (SSOT minimo)

## Politica ativa: quatro vetos de mercado

`orchestrator.execution.four_market_vetoes=true` substitui a lista de gates
de mercado legada abaixo. Sao quatro categorias, nao uma cota de quatro ciclos:

| Veto | Confluencia na ultima vela fechada |
|------|------------------------------------|
| `call_top_rejection` | RSI >= 0.75, BB %B >= 1 e pavio superior >= 45% do range |
| `call_down_continuation` | Corpo vendedor >= 80%, DI diff <= -0.15, tendencia e vela anterior PUT |
| `put_bottom_rejection` | RSI <= 0.25, BB %B <= 0 e pavio inferior >= 45% do range |
| `put_up_continuation` | Corpo comprador >= 80%, DI diff >= 0.15, tendencia e vela anterior CALL |

RSI e DI diff sao normalizados; OHLC usa precos absolutos. Candle ausente ou
invalido nao cria setup. Protecoes tecnicas de dados/deploy e caps permanecem
externas. `neg_edge`/`min_edge` sao bloqueios economicos, nao um quinto veto de
mercado. Os vetos extremos nao recebem waiver por PEND ou FLIP.

`market_direction_trigger=true` reavalia o lado antes do gate economico.
Um extremo contra o lado atual propoe o oposto, mas so o aceita se a
probabilidade calibrada existente sustentar `EV > min_edge_execute` e nao
houver extremo contrario ao candidato. EV = p(lado) * (1 + payout) - 1.
Registra `market_trigger_status`, `market_trigger_candidate`, setup e edge.
Nao estima uma nova probabilidade a partir do desenho do candle.

Limitacao deliberada: um TCN CALL com P(CALL)=0.52 nao autoriza PUT; P(PUT)=0.48.
Se o lado inicial ja e o argmax do TCN, o gatilho nao o inverte contra essa
distribuicao. Pode reconciliar um lado previamente alterado com o modelo.
Inversao independente de TCN exige outro estimador validado fora da amostra;
essa vantagem ainda nao foi demonstrada. Ativacao funcional nao e validacao
historica de rentabilidade. A lista abaixo descreve o fallback legado quando
`four_market_vetoes=false`.

Hot path vivo:

1. SKIP tecnico: `training` / `data` / `deploy` / `predict_error` / stop-win `EXEC_PAUSE`
2. SKIP sinal: `neg_edge` se Cal Edge ≤ 0 (`skip_neg_edge` **true**); waived somente com PEND ≥ `material_pending_min`; `acc_floor` so quando `skip_below_soft_min_acc` **true** (ops **false**); `counter_trend_unconfirmed` quando o lado contraria a tendência sem edge ≥ **0.08** (FLIP e anti-trend-lock não são vetados); `trend_discord` (tendencia e vela M5 discordam simultaneamente em EXPLORE); confluencia de mercado senior (`skip_exhaustion`, `skip_chop_congestion`, `skip_directional_momentum_discord`, `skip_two_bar_counter_trend`) e salvaguardas de price action/fluxo (`skip_wick_rejection`, `skip_climactic_blowoff`, `skip_opposing_marubozu`, `skip_adverse_tick_flow`) bloqueiam a entrada: nunca escolhem CALL/PUT.
3. TCN decide CALL/PUT (Cal ≥ 0.5 → CALL)
4. Anti-loss loss-clf = FLIP por `p_eff` apos auto_learn e `n_train >= 1` (young/mature 0.58; saida seed live N=2; ancora TCN) + **anti-trend-lock** ativo pos-loss
5. SCALE adapt **off** (`adapt_retract_enabled` **false**); `skip_doji` / `skip_exec_vs_candle` / `skip_scale_candle_discord` **false**
6. `invert_exec_side` **false**
7. Kelly + SIDE_EQ sizing (META **nao** soft Kelly)

**Proibido:** quality gate amplo arbitrario / Soft Kelly do loss-clf ou META / HARD SKIP loss-clf descalibrado / fusao EV / `cal_soft_edge` / SCALE adapt de lado sem trigger. IND RSI/ADX/HURST fora do log salvo em salvaguardas ativas.

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
