# Doutrina de trading para o LLM (Aether)

O LLM/Cursor e **copiloto de engenharia e auditoria**. Nao decide CALL/PUT em runtime.

Decisao live: TCN (Cal) → **unica FLIP** do `aether-loss-classifier` se auto_learn e `p_eff` >= **0.55** (shrink se N baixo; tape so telemetria) → Kelly + SIDE_EQ sizing. SKIP tecnico: treino/dados/deploy/predict/stop-win. Nenhum outro modulo pode inverter ordem.

**Removido:** HARD SKIP por loss-clf, Soft Kelly do loss-clf, fusao EV, signal_skip soft, micro/regime/vol/exhaust/neg_edge, anti-loss EMA/RSI, `invert_exec_side`.

Universo: **1HZ75V** M5 (contrato **5 m**; label N=1; ciclo **300 s**; payout **0.85**; stop-win **4.31%** Single-Strike).

## Nunca propor

1. `force_trade_every_cycle=true` como fix de EXEC_EMPTY
2. Reabrir quality gate amplo / `signal_skip` multi-gate / Soft do loss-clf / HARD SKIP no lugar do FLIP no piso
3. Revenge sizing apos LOSS
4. Remover caps, settlement ZSET ou timeouts “temporariamente”
5. Julgar mudanca so pelo P&L de poucos ciclos

## Sempre fazer

1. Distinguir EXPLORE vs RECOVER; `cover_enabled` **false**; piso **1%**
2. SKIP tecnico = processo ok quando coerente; FLIP `pe>=0.55` (apos auto_learn) e processo esperado
3. Evidencia: `live_n`, Cal/Edge, `val_accuracy`, telemetria `LOSS_CLF` (`p=` / `pe=` / `floor=`) e `QUALITY` pos-settle se houve FLIP
4. Pos-LOSS → container loss-clf `/learn`

## Diagnostico de log

CLUSTER → SCALE (vision) → `[GATES] || LOSS_CLF` (FLIP|OK|off) → KELLY → EXEC → RESOLVED.

Catalogo: [`engineering-indicator-gates.md`](engineering-indicator-gates.md). Playbook: [`binary-senior-playbook.md`](binary-senior-playbook.md).
