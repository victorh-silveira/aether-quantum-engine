# Playbook trader senior — binarias M5 (`1HZ75V`; OHLC 300s)

Postura: TCN **14D** decide CALL/PUT; **unica inversao de ordem** = **FLIP por p_eff** do `aether-loss-classifier` **so apos auto_learn** (nao bootstrap). Young (`n_train < 32`): shrink + piso **0.55**. Mature: piso **0.55**. Tape so telemetria. SKIP tecnico = treino/dados/deploy/broker/stop-win. Proibido SKIP/gate novo, Soft Kelly do loss-clf, HARD SKIP no piso, qualquer outro flip.

Universo: **1HZ75V** M5 (contrato **5 m**; label N=1; ciclo **300 s**).

Hierarquia: TCN → LOSS_CLF FLIP (se elegivel e p_eff >= piso) → Kelly / SIDE_EQ sizing → EXEC.

Telemetria: uma linha `[GATES] || LOSS_CLF: FLIP|OK|off …`. Pos-settle de FLIP: `LOSS_CLF || QUALITY flip=1 young= hit= p= pe= n=`.

Catalogo: [`engineering-indicator-gates.md`](engineering-indicator-gates.md).

## Quando operar

| Lado | Condicoes |
|------|-----------|
| CALL / PUT | TCN resolve lado; sem FLIP elegivel; sem SKIP tecnico |
| FLIP | auto_learn; `p_eff >= 0.55` → oposto do TCN |
| SKIP tecnico | `training` / `data` / `deploy` / `predict_error` / stop-win |

## Catalogo SKIP

| Razao | Significado |
|-------|-------------|
| tecnico | treino/dados/deploy/predict/stop-win |

Ver doutrina [`llm-trading-doctrine.md`](llm-trading-doctrine.md) e [`engineering-settings-ssot.md`](engineering-settings-ssot.md).
