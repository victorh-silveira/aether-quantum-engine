---
name: aether-session-review
description: >-
  Revisa sessoes live, logs do motor Aether e mudancas de risco/execucao com
  checklist PlayBook alinhada a doutrina LLM. Use when analyzing engine logs,
  EXEC_EMPTY, Kelly/recovery PRs, session post-mortems, or when the user mentions
  doutrina, PlayBook, ou revisao de sessao.
---

# Revisao de sessao / risco Aether

Ler `docs/llm-trading-doctrine.md` antes de concluir. LLM nao decide trade; avalia processo.

## Quando usar

- Usuario cola logs CLUSTER / SCALE / EXEC / KELLY / RISK
- PR ou diff em `execution_*`, `domain/risk`, `sample_size_policy`, `settings.json` de risco
- Pedido de pos-mortem ou “por que perdemos”

## Pre-trade (PlayBook)

1. Qual setup nomeado? (ex.: TCN resolve lado + Kelly; soft SIDE_EQ / scale_vision)
2. Qual bloqueio tecnico explicito? (`training`/`data`/`deploy`/`predict_error`)
3. Explore ou recover? Ha pending/linear? `scale_force_explore` ou `RECOVERY_INFEASIBLE`/`recovery_force_explore` bloquearam RECOVER/DAL? Pending material com `pending_waives_scale_explore` deve liberar soft cover.
4. Hipotese falsificavel da mudanca de knob (se houver)?

## During (leitura de log)

Ordem obrigatoria:

1. CLUSTER — Prob / Cal / Margin / Edge (telemetria); TF tipicamente micro **M1** (60 s)
2. SCALE — MACRO/MICRO/MINI/MILI + `tape`/`adapted` (adaptacao sob raw_extreme; soft Kelly; sem SKIP por escala)
3. EXEC / EMPTY / PAUSE — `gate_reason` tecnico ou Kelly; SIDE_EQ / scale = soft sizing (nao SKIP)
4. RESOLVED / RISK — pending, linear, pnl_sess

Marcar cada ciclo como: **processo ok** | **processo falhou** | **inconclusivo (N baixo)**.

Notas: `raw_extreme` mantem Cal para Kelly (nao e override MACRO TF). Escopo 1: sem veto de sinal / sem SKIP por escala.

## Pos-mortem (9 perguntas)

1. Taleb: confundimos streak com edge?
2. Mlodinow: N suficiente para a conclusao?
3. Ellenberg: taxa-base / ACC / Bayes respeitados? Comparar taxa-base do **lado no treino** (`label_call_frac`) vs distribuicao live CALL/PUT — vies de treino ≠ motivo para rearmar quality gate.
4. Duke: julgamos processo ou so P&L?
5. Bernstein: risco estava limitado (caps, pending)? SIDE_EQ / scale soft Kelly vs SKIP indevido?
6. Douglas: houve revenge sizing ou troca de regras mid-session?
7. PlayBook: setup e bloqueio tecnico estavam escritos?
8. Murphy: TA substituiu a TCN ou so filtrou telemetria (SCALE)?
9. LTCM: algum fail-safe tecnico (deploy/ACC/caps) foi removido?

## Saida esperada

Resposta curta em PT-BR:

- Veredito do processo (nao “sorte”)
- Ciclos problematicos com `gate_reason` tecnico / Cal / Edge / SCALE discord
- Acoes: manter knobs | ajuste minimo nomeado | retreino se ACC estruturalmente baixo
- Nunca recomendar `force_trade_every_cycle=true` como correcao
