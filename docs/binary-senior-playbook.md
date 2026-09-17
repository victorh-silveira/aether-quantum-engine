# Playbook trader senior — binarias M5 (`1HZ75V`; OHLC 300s)

Postura: TCN **14D** decide CALL/PUT; **FLIP** loss-clf por `p_eff` **apos auto_learn** (exit live **4**, `n_train>=4`). Young/mature pe>=**0.58**. Sem SCALE adapt / vela / META soft Kelly. SKIP tecnico = treino/dados/deploy/broker/stop-win / cooldown / pausa. Sem `cal_soft_edge` / quality gate / Soft Kelly do loss-clf.

Universo: **1HZ75V** M5 (contrato **5 m**; label N=1; ciclo **300 s**).

Hierarquia: TCN → LOSS_CLF FLIP → SKIP `neg_edge` → Kelly / SIDE_EQ → EXEC.

Telemetria ciclo: CLUSTER → GATES/LOSS_CLF → KELLY → EXEC → RESOLVED. Em bootstrap: `blocked=bootstrap boot=N/4`; `blocked=flip_min_n` se `n<4`. Edge CLUSTER = EV vs BE.

Algoritmo mental: `lado=TCN(Cal)` → FLIP se pe no piso → Kelly.

Catalogo: [`engineering-indicator-gates.md`](engineering-indicator-gates.md).

## Quando operar

| Lado | Condicoes |
|------|-----------|
| CALL / PUT | TCN resolve lado; sem FLIP; sem SKIP tecnico |
| FLIP | auto_learn; `n_train>=4`; `p_eff` no piso → oposto do TCN |
| SKIP `neg_edge` | Cal Edge ≤ 0 em EXPLORE (waive com PEND material) |
| SKIP tecnico | `training` / `data` / `deploy` / `predict_error` / stop-win / cooldown / pausa |

## Catalogo SKIP

| Razao | Significado |
|-------|-------------|
| tecnico | treino/dados/deploy/predict/stop-win / cooldown pos-LOSS / pausa de sessao |
| `neg_edge` | Edge TCN ≤ 0 em EXPLORE |
