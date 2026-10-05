# Playbook trader senior — binarias M5 (`1HZ75V`; OHLC 300s)

Atualizacao operacional: com `four_market_vetoes=true`, a politica ativa e
[quatro vetos extremos](engineering-indicator-gates.md), dois por lado.
O gatilho `market_direction_trigger` pode reconciliar direcao com a probabilidade
TCN existente somente se o candidato passar EV, haircut e margem sobre
break-even no payout observado. As regras
legadas de contra-tendencia descritas abaixo so valem no fallback desligado.

Postura: TCN **14D** decide CALL/PUT; **FLIP** loss-clf por `p_eff` **apos auto_learn** (exit live **2**, `n_train>=1`). Young pe>=**0.58** / mature pe>=**0.70** (`flip_young_shrink` **0.50**). Sem SCALE adapt / vela / META soft Kelly. Nao ha bloqueio temporal pos-LOSS. Quatro vetos extremos de mercado e a cotacao final com EV positivo e margem de 0,01 sobre break-even protegem a compra. O regime e telemetrico; ruido/tendencia/conformal/compressao nao somam vetos no SSOT.

Universo: **1HZ75V** M5 (contrato **5 m**; label N=1; ciclo **300 s**).

Hierarquia: TCN → LOSS_CLF FLIP apenas com edge e margem do lado candidato → anti-trend-lock/Alpha Flip com o mesmo criterio → quatro vetos extremos e gatilho de reconciliacao → Kelly / SIDE_EQ → proposta com quote guard → EXEC.

Telemetria ciclo: CLUSTER → GATES/LOSS_CLF → KELLY → EXEC → RESOLVED. Em bootstrap: `blocked=bootstrap boot=N/2`; `blocked=flip_min_n` se `n<1`. Edge CLUSTER e estimativa previa; a cotacao final usa payout do broker. Inversao sem edge estimado registra rejeicao e preserva o lado anterior; a proposta sempre usa P(lado) calibrada, sem conviccao fixa de recuperacao.

Algoritmo mental: `lado=TCN(Cal)` → FLIP se pe no piso → Kelly.

Catalogo: [`engineering-indicator-gates.md`](engineering-indicator-gates.md).

## Quando operar

| Lado | Condicoes |
|------|-----------|
| CALL / PUT | TCN resolve lado; sem FLIP; sem SKIP tecnico |
| FLIP | auto_learn; `n_train>=1`; `p_eff` no piso → oposto do TCN |
| SKIP economico | Probabilidade ausente/invalida ou cotacao final sem EV e margem suficientes |
| SKIP tecnico | `training` / `data` / `deploy` / `predict_error` / stop-win |

## Catalogo SKIP

| Razao | Significado |
|-------|-------------|
| tecnico | treino/dados/deploy/predict/stop-win |
| `quote_edge_below_min` | EV da cotacao final abaixo do piso |
