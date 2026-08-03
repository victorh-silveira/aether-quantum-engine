---
name: aether-binary-senior
description: >-
  Avalia sessoes e mudancas de gate no estilo trader senior de opcoes binarias
  120s (CALL/PUT/SKIP). Use when analyzing CLUSTER/Cal/Edge logs, gate_reason,
  indicator conflicts, or when the user mentions playbook senior, SKIP, or
  binarias R_10.
---

# Playbook senior binario

Ler `docs/binary-senior-playbook.md` e `execution_senior_skip.py`.

## Checklist

1. Cal margin ≥ 0.05? Senao SKIP `cal_margin_floor` (processo ok)
2. ADX ≥ 0.16 e Hurst fora de 0.47–0.53?
3. `align_rsi_trend`: RSI/DI alinhados ao lado TCN?
4. Edge meta ≥ 0? ACC ≥ 0.53?
5. EXPLORE vs RECOVER — nao revenge sizing

## Saida

Veredito: **CALL elegivel** | **PUT elegivel** | **SKIP justificado** | **bad fill** (processo falhou).
Nunca recomendar `force_trade_every_cycle=true`.
