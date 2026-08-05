---
name: aether-binary-senior
description: >-
  Avalia sessoes live no estilo trader senior de opcoes binarias 30s
  (CALL/PUT/SKIP tecnico; micro OHLC 60s hibrido). Use when analyzing CLUSTER/Cal/Edge logs, gate_reason
  tecnico, or when the user mentions playbook senior, SKIP, or binarias R_10.
---

# Playbook senior binario

Ler `docs/binary-senior-playbook.md`.

## Checklist (pos-escopo 1)

1. Bloqueio tecnico? (`training`/`data`/`deploy`/`predict_error`) — senao TCN segue
2. Candidato `execution_candidate_ready`? Cal/Edge sao telemetria, nao veto
3. ACC/deploy de treino ≥ 0.53 quando o tema for modelo; checar `label_call_frac` / majority-collapse
4. Lado enviesado no live ≠ SKIP tecnico — SIDE_EQ e **soft Kelly** (`execution_side_eq_sizing`), nao veto de direcao
5. SCALE: par MINI, **retracao**, **explosao** ou **mili+tape** vs TCN; `micro=retract|explos|chop`; dampen/force EXPLORE; Kelly `kelly_p_floor`
6. `raw_extreme` ≠ MACRO TF: Cal nao e substituido por raw; limiares `tcn_macro_*_override` so limiam raw
7. Kelly/caps — `EXEC_PAUSE` so `stop_win` / banca; **sem** `kelly_no_edge`; explore fino usa `neutral_bankroll_pct` (~0.25%), nao `$1`
8. EXPLORE vs RECOVER — nao revenge sizing; discordance/adapt forca EXPLORE Kelly (sem DAL_Ln) **exceto** quando `pending_waives_scale_explore` e pending material (soft cover sob teto); `RECOVERY_INFEASIBLE`/`infeasible_force_explore` tambem forca EXPLORE (sem DAL no teto)

## Saida

Veredito: **CALL elegivel** | **PUT elegivel** | **SKIP tecnico** | **bad fill** (processo falhou).
Nunca recomendar `force_trade_every_cycle=true`.
Nunca propor rearmar quality gate / veto de sinal por streak PUT/CALL.
