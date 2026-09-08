# Catalogo de gates (SSOT minimo)

Hot path vivo apos purge:

1. SKIP tecnico: `training` / `data` / `deploy` / `predict_error` / stop-win `EXEC_PAUSE`
2. TCN decide CALL/PUT
3. Anti-loss = HARD SKIP so por `P_LOSS` do container `aether-loss-classifier` (`gate_reason=loss_clf`) quando `p_loss >= hard_p_loss_floor` (**0.90**)
4. Kelly + SIDE_EQ sizing (nao e gate de direcao)

**Removido:** signal_skip soft, fusao EV, SCALE adapt, micro protect, regime squeeze, chop soft, vol/exhaust, neg_edge, anti-loss legado (EMA/RSI/vela), FLIP loss-clf, invert_exec_side.

Telemetria: uma linha `[GATES] || LOSS_CLF: HARD|OK|off …`.

Protocolo: nao reabrir quality gate amplo; gate novo so com mandato explicito e SSOT nomeado.

Refs: playbook, `engineering-settings-ssot.md`, rule `aether-execution-gates.mdc`, skill `aether-binary-senior`.
