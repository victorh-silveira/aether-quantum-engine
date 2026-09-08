---
name: aether-binary-senior
description: >-
  Avalia sessoes live M5 1HZ75V: TCN + HARD SKIP por P_LOSS do loss-classifier.
  Use when analyzing CLUSTER/Cal/Edge logs, gate_reason loss_clf, or playbook senior.
---

# Playbook senior binario (`1HZ75V` / M5)

Ler `docs/binary-senior-playbook.md` e `docs/engineering-indicator-gates.md`.

Pipeline: TCN 14D → LOSS_CLF HARD se `p_loss >= 0.90` → Kelly. Sem fusao/micro/regime/vol/exhaust/neg_edge/FLIP.

## Checklist

1. SKIP tecnico?
2. `LOSS_CLF HARD` / `gate_reason=loss_clf`?
3. Cal/Edge telemetria; Kelly Single-Strike so se ALLOW
4. EXEC_EMPTY tecnico ou loss_clf = processo ok quando coerente

## Proibido

- Reabrir quality gate / signal_skip multi-gate / FLIP / Soft_SIZE do loss-clf
- force_trade como fix de EMPTY
