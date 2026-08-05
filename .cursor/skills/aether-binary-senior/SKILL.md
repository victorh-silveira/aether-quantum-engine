---
name: aether-binary-senior
description: >-
  Avalia sessoes live no estilo trader senior de opcoes binarias 30s
  (CALL/PUT/SKIP tecnico ou signal_skip 1.1; micro OHLC 60s hibrido). Use when analyzing CLUSTER/Cal/Edge logs, gate_reason
  tecnico, or when the user mentions playbook senior, SKIP, or binarias R_10.
---

# Playbook senior binario

Ler `docs/binary-senior-playbook.md`.

## Checklist (escopo 1.1)

1. Bloqueio tecnico? (`training`/`data`/`deploy`/`predict_error`) — senao segue TCN/SCALE
2. Catálogo `signal_skip`? `mini_pair_oppose` / `cal_margin` (waive pending) — senao candidato segue
3. ACC/deploy de treino ≥ 0.53 quando o tema for modelo; checar `label_call_frac` / majority-collapse
4. Lado enviesado no live ≠ motivo para quality gate amplo — SIDE_EQ e **soft Kelly**
5. SCALE: par MINI, **retracao**, **explosao** ou **mili+tape** vs TCN; dampen/force EXPLORE; Kelly `kelly_p_floor`
6. `raw_extreme` ≠ MACRO TF: Cal nao e substituido por raw
7. Kelly/caps — explore fino `neutral_bankroll_pct` (~0.25%)
8. EXPLORE vs RECOVER — `pending_waives_scale_explore`; sem revenge sizing
9. Pos-LOSS: telemetria/veto via `aether-loss-classifier` (log `LOSS_CLF`); nao quality gate amplo

## Saida

Veredito: **CALL elegivel** | **PUT elegivel** | **SKIP tecnico** | **SKIP sinal 1.1** | **bad fill**.
Nunca recomendar `force_trade_every_cycle=true`.
Nunca propor quality gate amplo (Hurst/ADX/RSI) por streak.
