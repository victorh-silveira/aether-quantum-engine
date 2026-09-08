# Playbook trader senior — binarias M5 (`1HZ75V`; OHLC 300s)

Postura: TCN **14D** decide CALL/PUT; unico filtro vivo de risco de sinal = **HARD SKIP por P_LOSS** do `aether-loss-classifier` (`p_loss >= 0.90` → `gate_reason=loss_clf`). SKIP tecnico = treino/dados/deploy/broker/stop-win. Sem fusao EV, sem micro/regime/vol/exhaust/neg_edge, sem anti-loss EMA/RSI, sem FLIP.

Universo: **1HZ75V** M5 (contrato **5 m**; label N=1; ciclo **300 s**).

Hierarquia: TCN → LOSS_CLF HARD (P_LOSS) → Kelly / SIDE_EQ sizing → EXEC.

Catalogo: [`engineering-indicator-gates.md`](engineering-indicator-gates.md).

## Quando operar

| Lado | Condicoes |
|------|-----------|
| CALL / PUT | TCN resolve lado; `p_loss` abaixo do piso HARD; sem SKIP tecnico |
| SKIP tecnico | `training` / `data` / `deploy` / `predict_error` / stop-win |
| HARD `loss_clf` | `p_loss >= hard_p_loss_floor` (**0.90**) |

## Catalogo SKIP

| Razao | Significado |
|-------|-------------|
| tecnico | treino/dados/deploy/predict/stop-win |
| `loss_clf` | P_LOSS alto no sidecar loss-classifier |

Ver doutrina [`llm-trading-doctrine.md`](llm-trading-doctrine.md) e [`engineering-settings-ssot.md`](engineering-settings-ssot.md).
