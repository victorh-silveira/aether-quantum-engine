# Doutrina de trading para o LLM (Aether)

O LLM/Cursor e **copiloto de engenharia e auditoria**. Nao decide CALL/PUT em runtime.

Decisao live: TCN (Cal vs 0.5) → **FLIP** loss-clf se auto_learn e `p_eff` no piso → SCALE retract/explos adapt (ultima palavra; pode desfazer FLIP) → Kelly (META edge ≤ 0 = soft Kelly). SKIP tecnico: treino/dados/deploy/predict/stop-win / cooldown / pausa. Sem `cal_soft_edge` / quality gate.

**Removido:** HARD SKIP por loss-clf, Soft Kelly do loss-clf, fusao EV, signal_skip soft, micro/regime/vol/exhaust/neg_edge, anti-loss EMA/RSI, `invert_exec_side`, `cal_soft_edge`.

Universo: **1HZ75V** M5 (contrato **5 m**; label N=1; ciclo **300 s**; payout **0.85**; stop-win **4.31%** Single-Strike).

## Nunca propor

1. `force_trade_every_cycle=true` como fix de EXEC_EMPTY
2. Reabrir quality gate amplo / `signal_skip` multi-gate / Soft do loss-clf / HARD SKIP no lugar do FLIP no piso / trava por Edge
3. Revenge sizing apos LOSS
4. Remover caps, settlement ZSET ou timeouts “temporariamente”
5. Julgar mudanca so pelo P&L de poucos ciclos

## Sempre fazer

1. Distinguir EXPLORE vs RECOVER; `cover_enabled` **false**; piso **1%**
2. SKIP tecnico = processo ok quando coerente; FLIP young `pe>=0.55` / mature `pe>=0.58` (apos auto_learn) e processo esperado
3. Evidencia: `live_n`, Cal/Edge (EV vs BE), `val_accuracy`, telemetria `LOSS_CLF` (`p=` / `pe=` / `floor=` / `boot=N/4`) e `QUALITY` pos-settle se houve FLIP
4. Pos-LOSS → container loss-clf `/learn`

## Diagnostico de log

CLUSTER → SCALE (vision) → `[GATES] || LOSS_CLF` → SCALE adapt (last) → KELLY → EXEC → RESOLVED.

Leitura critica:

- `adapted=1` + `why=retract_vs_tcn` = SCALE virou contra o TCN/EXEC pos-FLIP pela retracao.
- `adapted=1` + `why=explos_vs_tcn` = SCALE virou pela explosao com mi+mili alinhados (mili discordante = sem adapt).
- `adapted=1` + `why=tape_vs_tcn` = fita forte (`tape_strong`) oposta ao TCN/EXEC; discordance sozinha nao adapta (`tape_not_strong`).
- `adapted=1` + `why=retract_holds` / `explos_holds` / `tape_holds` = adapt desfez um FLIP.
- Edge CLUSTER = EV vs BE; negativo sozinho nao skipa.
- `META: applied=1 edge≤0` = soft Kelly (`meta_soft_kelly`; forte se edge ≤ -0.5 → factor **0.40**); Soft_SIZE **nao** re-eleva stake.
- `META: sat=1` com Cal Edge ≤ 0 = soft Kelly (`meta_sat_vs_neg_cal`).
- KELLY: `meta_soft=1 strong=0|1 kscale=` quando soft ativo.
- `[CANDLE]` same-cycle = vela fechada; ciclo seguinte = janela do contrato anterior.

Catalogo: [`engineering-indicator-gates.md`](engineering-indicator-gates.md). Playbook: [`binary-senior-playbook.md`](binary-senior-playbook.md).
