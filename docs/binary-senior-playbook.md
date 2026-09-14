# Playbook trader senior — binarias M5 (`1HZ75V`; OHLC 300s)

Postura: TCN **14D** decide CALL/PUT; **FLIP** loss-clf por `p_eff` **apos auto_learn** (exit live **4**, `n_train>=8`); SCALE **retract/explos/tape** last **exceto** FLIP sticky sem SKIP. Young/mature pe>=**0.58**. Tape/IND = telemetria. SKIP tecnico = treino/dados/deploy/broker/stop-win / cooldown / pausa. Sem `cal_soft_edge` / quality gate / Soft Kelly do loss-clf.

Universo: **1HZ75V** M5 (contrato **5 m**; label N=1; ciclo **300 s**).

Hierarquia: TCN → LOSS_CLF FLIP → SCALE retract/explos adapt (last; FLIP sticky) → Kelly / SIDE_EQ → EXEC (META edge ≤ 0 comprime Kelly via `soft_veto_score_factor`, sem SKIP).

Telemetria: `SCALE: … adapted=0|1 why=retract_vs_tcn|explos_vs_tcn|tape_vs_tcn|flip_holds|explos_edge_firm …`. `flip_holds` = manteve FLIP; `explos_edge_firm` = Edge TCN > **0.05**. `META: applied=1 edge=-1` → soft Kelly (`meta_soft_kelly`). Em bootstrap: `blocked=bootstrap boot=N/4`; `blocked=flip_min_n` se `n<8`. Edge CLUSTER = EV vs BE. `[CANDLE]` same-cycle = vela fechada; ciclo seguinte = janela do contrato anterior.

Catalogo: [`engineering-indicator-gates.md`](engineering-indicator-gates.md).

## Quando operar

| Lado | Condicoes |
|------|-----------|
| CALL / PUT | TCN resolve lado; sem FLIP; sem SCALE adapt; sem SKIP tecnico |
| FLIP | auto_learn; `n_train>=8`; `p_eff` no piso → oposto do TCN (ancora TCN) |
| SCALE adapt | sem FLIP: retract/explos (mi+mili; explos so Edge ≤**0.05**) ou tape **forte** (tambem apos `mili_mismatch`/`explos_edge_firm`); com FLIP → `flip_holds` |
| META soft Kelly | META edge ≤ 0 (ou sat vs Cal≤0) → soft Kelly; forte edge≤-0.5 factor **0.40**; **nao** re-eleva stake com piso Soft_SIZE 2.5%; sem SKIP |
| SKIP tecnico | `training` / `data` / `deploy` / `predict_error` / stop-win / cooldown / pausa |

## Catalogo SKIP

| Razao | Significado |
|-------|-------------|
| tecnico | treino/dados/deploy/predict/stop-win / cooldown pos-LOSS / pausa de sessao |

Ver doutrina [`llm-trading-doctrine.md`](llm-trading-doctrine.md) e [`engineering-settings-ssot.md`](engineering-settings-ssot.md).
