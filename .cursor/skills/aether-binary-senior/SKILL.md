---
name: aether-binary-senior
description: >-
  Avalia sessoes live M5 1HZ75V: TCN + FLIP por P_LOSS do loss-classifier.
  Use when analyzing CLUSTER/Cal/Edge logs, LOSS_CLF FLIP, or playbook senior.
---

# Playbook senior binario (`1HZ75V` / M5)

Ler `docs/binary-senior-playbook.md` e `docs/engineering-indicator-gates.md`.

Pipeline: TCN 14D → LOSS_CLF FLIP se auto_learn e pe>=0.55 → Kelly. Sem HARD SKIP/fusao/micro/regime/vol/exhaust/neg_edge. Indicadores = feature (catalogo de gates fechado). Pos-settle: `QUALITY` se houve FLIP. Indicadores = feature (catalogo de gates fechado). Pos-settle: `LOSS_CLF || QUALITY`.

## Checklist

1. SKIP tecnico?
2. `LOSS_CLF FLIP` (lado invertido vs TCN)?
3. Cal/Edge telemetria; Kelly Single-Strike so se ALLOW
4. EXEC_EMPTY tecnico = processo ok quando coerente

## Proibido

- Reabrir quality gate / signal_skip / Soft_SIZE do loss-clf / HARD SKIP no piso
- force_trade como fix de EMPTY
