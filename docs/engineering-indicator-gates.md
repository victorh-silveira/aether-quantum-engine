# Catalogo de gates (SSOT minimo)

Hot path vivo:

1. SKIP tecnico: `training` / `data` / `deploy` / `predict_error` / stop-win `EXEC_PAUSE`
2. TCN decide CALL/PUT
3. Anti-loss = **unica inversao** CALL↔PUT so por `p_eff` do `aether-loss-classifier` **apos auto_learn**: young (`n < 32`) `p_eff >= 0.55` apos shrink; mature `p_eff >= 0.55`; tape so telemetria; executa o lado invertido
4. Kelly + SIDE_EQ sizing (nao e gate de direcao; **nao flipa**)

**Proibido:** qualquer flip que nao seja p_eff no piso; HARD SKIP por loss-clf; Soft Kelly do loss-clf; signal_skip; fusao EV; SCALE adapt; micro/regime/vol/exhaust/neg_edge; anti-loss EMA/RSI; invert_exec_side.

Telemetria: uma linha `[GATES] || LOSS_CLF: FLIP|OK|off …` (`p=` cru, `pe=` efetivo se diferir, `floor=` piso de decisao).

Protocolo: nao reabrir quality gate amplo; gate novo so com mandato explicito e SSOT nomeado.

Refs: playbook, `engineering-settings-ssot.md`, rule `aether-execution-gates.mdc`, skill `aether-binary-senior`.
