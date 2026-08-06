---
name: aether-binary-senior
description: >-
  Avalia sessoes live no estilo trader senior de opcoes binarias M15
  (CALL/PUT/SKIP tecnico ou signal_skip 1.1; OHLC 900s) no indice
  OTC_SPC (S&P 500 OTC). Use when analyzing CLUSTER/Cal/Edge logs, gate_reason
  tecnico, or when the user mentions playbook senior, SKIP, SP500, M15, ou binarias OTC_SPC.
---

# Playbook senior binario (`OTC_SPC` / M15)

Ler `docs/binary-senior-playbook.md` e `docs/deriv-indices-algorithm.md`.

Universo: **S&P 500 OTC** (`OTC_SPC`) — **somente M15** (contrato 15 m; ciclo/micro 900 s).

## Checklist (escopo 1.1)

1. Bloqueio tecnico? (`training`/`data`/`deploy`/`predict_error`) — senao segue TCN/SCALE
2. Catalogo `signal_skip`? mini/cal = soft Kelly — senao candidato segue (sem flip pos-LOSS)
3. ACC/deploy de treino >= 0.53 quando o tema for modelo; checar `label_call_frac` / majority-collapse
4. Lado enviesado no live != motivo para quality gate amplo — SIDE_EQ e **soft Kelly**
5. SCALE: **majority_votes** (tape/mili/RSI vs TCN); par MINI, **retracao**, **explosao** ou **mili+tape**; dampen/force EXPLORE; Kelly `kelly_p_floor`
6. Soft Kelly: mini/cal/loss-clf; loss-clf graduado (**0.55->0.40**) + teto soft **2%**; pending -> cover **100%**
7. `raw_extreme` != MACRO TF: Cal nao e substituido por raw
8. Kelly/caps — explore piso `neutral_bankroll_pct` (**2%**); RECOVER `cover_multiple` **2**; teto **5%**
9. EXPLORE vs RECOVER — `pending_waives_scale_explore`; sem revenge sizing
10. Pos-LOSS: telemetria/soft via `aether-loss-classifier` (log `LOSS_CLF || SOFT`); retrain LOSS n>=2; lado permanece no TCN

## Saida

Veredito: **CALL elegivel** | **PUT elegivel** | **SKIP tecnico** | **soft sinal 1.1** | **bad fill**.
Nunca recomendar `force_trade_every_cycle=true`.
Nunca propor quality gate amplo (Hurst/ADX/RSI) por streak.
