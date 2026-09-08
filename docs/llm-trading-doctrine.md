# Doutrina de trading para o LLM (Aether)

O LLM/Cursor e **copiloto de engenharia e auditoria**. Nao decide CALL/PUT em runtime.

Decisao live: TCN (Cal) → **HARD SKIP** do `aether-loss-classifier` se `p_loss >= hard_p_loss_floor` (**0.90**, `gate_reason=loss_clf`) → Kelly + SIDE_EQ sizing. SKIP tecnico: treino/dados/deploy/predict/stop-win.

**Removido da doutrina viva:** fusao EV, signal_skip soft, micro/regime/vol/exhaust/neg_edge, anti-loss EMA/RSI, FLIP CALL↔PUT, Soft Kelly do loss-clf, `invert_exec_side`.

Universo: **1HZ75V** M5 (contrato **5 m**; label N=1; ciclo **300 s**; payout **0.85**; stop-win **4.31%** Single-Strike).

## Nunca propor

1. `force_trade_every_cycle=true` como fix de EXEC_EMPTY
2. Reabrir quality gate amplo / `signal_skip` multi-gate / FLIP / Soft do loss-clf
3. Revenge sizing apos LOSS
4. Remover caps, settlement ZSET ou timeouts “temporariamente”
5. Julgar mudanca so pelo P&L de poucos ciclos

## Sempre fazer

1. Distinguir EXPLORE vs RECOVER; `cover_enabled` **false**; piso **1%**
2. HARD `loss_clf` e SKIP tecnico = processo ok quando coerente
3. Evidencia: `live_n`, Cal/Edge, `val_accuracy`, telemetria `LOSS_CLF`
4. Pos-LOSS → container loss-clf `/learn` (sem FLIP no hot path)

## Diagnostico de log

CLUSTER → SCALE (vision) → `[GATES] || LOSS_CLF` → KELLY → EXEC → RESOLVED.

Catalogo: [`engineering-indicator-gates.md`](engineering-indicator-gates.md). Playbook: [`binary-senior-playbook.md`](binary-senior-playbook.md).
