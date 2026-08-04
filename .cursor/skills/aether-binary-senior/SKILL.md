---
name: aether-binary-senior
description: >-
  Avalia sessoes live no estilo trader senior de opcoes binarias 120s
  (CALL/PUT/SKIP tecnico). Use when analyzing CLUSTER/Cal/Edge logs, gate_reason
  tecnico, or when the user mentions playbook senior, SKIP, or binarias R_10.
---

# Playbook senior binario

Ler `docs/binary-senior-playbook.md`.

## Checklist (pos-escopo 1)

1. Bloqueio tecnico? (`training`/`data`/`deploy`/`predict_error`) — senao TCN segue
2. Candidato `execution_candidate_ready`? Cal/Edge sao telemetria, nao veto
3. ACC/deploy de treino ≥ 0.53 quando o tema for modelo; checar `label_call_frac` / majority-collapse
4. Lado enviesado no live ≠ SKIP tecnico — SIDE_EQ e **soft Kelly** (`execution_side_eq_sizing`), nao veto de direcao
5. SCALE: `SCALE || … tape=… adapted=` e IND `SCALE: tcn=… tape=…` — adapta com **par MINI prev+curr** alinhado e (`raw_extreme` ou fita forte); discord/adapt amortece Kelly / corta DAL
6. `raw_extreme` ≠ MACRO TF: Cal nao e substituido por raw; limiares `tcn_macro_*_override` so limiam raw
7. Kelly/caps — `EXEC_PAUSE` e sizing, nao veto de direcao
8. EXPLORE vs RECOVER — nao revenge sizing; discordance/adapt forca EXPLORE Kelly (sem DAL_Ln)

## Saida

Veredito: **CALL elegivel** | **PUT elegivel** | **SKIP tecnico** | **bad fill** (processo falhou).
Nunca recomendar `force_trade_every_cycle=true`.
Nunca propor rearmar quality gate / veto de sinal por streak PUT/CALL.
