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
3. ACC/deploy de treino ≥ 0.53 quando o tema for modelo
4. Kelly/caps — `EXEC_PAUSE` e sizing, nao veto de direcao
5. EXPLORE vs RECOVER — nao revenge sizing

## Saida

Veredito: **CALL elegivel** | **PUT elegivel** | **SKIP tecnico** | **bad fill** (processo falhou).
Nunca recomendar `force_trade_every_cycle=true`.
