# Playbook trader senior — binarias M5 (`1HZ75V`; OHLC 300s)

Atualizacao operacional: com `four_market_vetoes=true`, a politica ativa e
[quatro vetos extremos](engineering-indicator-gates.md), dois por lado.
O gatilho `market_direction_trigger` pode reconciliar direcao com a probabilidade
TCN existente, mas nao converte `neg_edge` em evidencia para inverter. As regras
legadas de contra-tendencia descritas abaixo so valem no fallback desligado.

Postura: TCN **14D** decide CALL/PUT; **FLIP** loss-clf por `p_eff` **apos auto_learn** (exit live **4**, `n_train>=4`). Young pe>=**0.58** / mature pe>=**0.70** (`flip_young_shrink` **0.50**). Sem SCALE adapt / vela / META soft Kelly. Não há bloqueio temporal pós-LOSS; lado contra a tendência exige edge ≥ **0.08**, exceto FLIP/anti-trend-lock. SKIP tecnico = treino/dados/deploy/broker/stop-win. Sem `cal_soft_edge` / quality gate / Soft Kelly do loss-clf.

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
| SKIP tecnico | `training` / `data` / `deploy` / `predict_error` / stop-win |

## Catalogo SKIP

| Razao | Significado |
|-------|-------------|
| tecnico | treino/dados/deploy/predict/stop-win |
| `neg_edge` | Edge TCN ≤ 0 em EXPLORE |
