---
name: aether-binary-senior
description: >-
  Avalia sessoes live M5 1HZ75V: TCN + FLIP loss-clf (sem SCALE adapt).
  Use when analyzing CLUSTER/Cal/Edge logs, LOSS_CLF FLIP, or playbook senior.
---

# Playbook senior binario (`1HZ75V` / M5)

Ler `docs/binary-senior-playbook.md` e `docs/engineering-indicator-gates.md`.

Pipeline: TCN 14D (CALL se Cal ≥0.5 senao PUT) → LOSS_CLF FLIP se auto_learn (exit **2**, `n_train>=1`), pe no piso configurado e candidato com EV e margem no payout observado → anti-trend-lock/Alpha Flip sob o mesmo criterio → `invert_exec_side` **false** → quatro vetos extremos e gatilho de reconciliacao sob o mesmo quote guard → Kelly / cover amort **1** → proposta com EV positivo e margem. Sem SCALE adapt / doji / exec_vs_candle / META soft Kelly.

## Checklist (auditoria por ciclo)

1. SKIP tecnico: treino/dados/checkpoint/predict/stop-win; SKIP sinal: quatro vetos extremos e cotacao final sem EV ou margem suficientes. Ruido/tendencia/conformal/compressao nao somam vetos no SSOT.
2. CLUSTER → lado TCN (Cal≥0.5 CALL) + Edge EV
3. `LOSS_CLF`: FLIP se pe no piso; `blocked=bootstrap|flip_min_n` → sem FLIP
4. KELLY/EXEC lado = TCN(+FLIP); RESOLVED valida mercado
5. Inversao rejeitada deve preservar o lado anterior; a proposta deve usar P(lado) calibrada e payout cotado, sem conviccao fixa de recuperacao.

## Proibido

- Reabrir SCALE adapt / Soft Kelly META ou loss-clf / HARD SKIP / `cal_soft_edge`
- force_trade como fix de EMPTY
- `invert_exec_side=true` sem mandato
