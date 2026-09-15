---
name: aether-binary-senior
description: >-
  Avalia sessoes live M5 1HZ75V: TCN + FLIP loss-clf + SCALE retract/explos last.
  Use when analyzing CLUSTER/Cal/Edge logs, LOSS_CLF FLIP, SCALE adapted, or playbook senior.
---

# Playbook senior binario (`1HZ75V` / M5)

Ler `docs/binary-senior-playbook.md` e `docs/engineering-indicator-gates.md`.

Pipeline: TCN 14D (CALL se Cal ≥0.5 senao PUT) → LOSS_CLF FLIP se auto_learn (exit **4**, `n_train>=8`) e pe no piso (**exceto** `blocked=candle_holds` se vela == TCN) → SCALE retract/explos/tape → **vela fechada** `candle_vs_tcn` (FLIP sticky) → Kelly (META edge ≤ 0 = soft Kelly). Sem `SKIP:NEUTRAL_ZONE`. Sem trava `cal_soft_edge`. Sem quality gate. Pos-settle: `LOSS_CLF || QUALITY` se houve FLIP loss-clf.

## Checklist (auditoria por ciclo)

1. SKIP tecnico: `training` / `data` / `deploy` / `predict_error` / stop-win / cooldown / pausa / WSS FAIL
2. CLUSTER → anotar lado TCN (Cal≥0.5 CALL) + Edge EV
3. CANDLE → `dir_vela` same-cycle vs TCN
4. `LOSS_CLF`: FLIP so se pe no piso e vela ≠ TCN; `blocked=bootstrap|flip_min_n|candle_holds` → sem FLIP; padrao c5 (FLIP com vela==TCN) = processo errado
5. SCALE last: regime (mi==mili; explos so Edge≤0.05) → tape_strong → candle last; `why=…`; com FLIP → `flip_holds`
6. META so sizing (`meta_soft_kelly`); IND RSI/ADX/HURST so telemetria
7. KELLY/EXEC lado = cascata; RESOLVED valida mercado, nao so processo

## Proibido

- Reabrir quality gate / signal_skip / Soft_SIZE do loss-clf / HARD SKIP / `cal_soft_edge`
- force_trade como fix de EMPTY
