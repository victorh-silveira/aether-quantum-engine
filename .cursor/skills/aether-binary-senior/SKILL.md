---
name: aether-binary-senior
description: >-
  Avalia sessoes live M5 1HZ75V: TCN + FLIP loss-clf + SCALE retract/explos last.
  Use when analyzing CLUSTER/Cal/Edge logs, LOSS_CLF FLIP, SCALE adapted, or playbook senior.
---

# Playbook senior binario (`1HZ75V` / M5)

Ler `docs/binary-senior-playbook.md` e `docs/engineering-indicator-gates.md`.

Pipeline: TCN 14D (CALL se Cal ≥0.5 senao PUT) → LOSS_CLF FLIP se auto_learn (exit **4**) e pe no piso → SCALE retract/explos adapt (last; pode desfazer FLIP) → Kelly (META edge ≤ 0 = soft Kelly). Sem `SKIP:NEUTRAL_ZONE`. Sem trava `cal_soft_edge`. Sem quality gate. Pos-settle: `LOSS_CLF || QUALITY` se houve FLIP loss-clf.

## Checklist

1. SKIP tecnico: `training` / `data` / `deploy` / `predict_error` / stop-win / cooldown pos-LOSS / pausa de sessao
2. `LOSS_CLF FLIP`? Se `blocked=bootstrap boot=N/4` → FLIP ainda off
3. SCALE last: `adapted=1` + `retract_vs_tcn` / `explos_vs_tcn` / `tape_vs_tcn` ou `*_holds` (retract/explos precisam mi+mili; tape precisa `tape_strong`) — nao e SKIP; `tape_not_strong` = discord sem adapt
4. META `edge≤0` ou `sat=1`+Cal Edge≤0 → soft Kelly (nao SKIP; nao flipa; nao re-eleva stake via Soft_SIZE 2.5%)
5. Cal/Edge = EV vs BE (telemetria); Edge negativo sozinho nao e skip
6. EXEC_EMPTY tecnico = processo ok quando coerente

## Proibido

- Reabrir quality gate / signal_skip / Soft_SIZE do loss-clf / HARD SKIP / `cal_soft_edge`
- force_trade como fix de EMPTY
