---
name: aether-session-review
description: >-
  Revisa sessoes live, logs do motor Aether e mudancas de risco/execucao com
  checklist PlayBook (pre-trade, during, pos-mortem) alinhada aos 9 livros da
  doutrina LLM. Use when analyzing engine logs, EXEC_EMPTY, Kelly/recovery PRs,
  gate changes, session post-mortems, or when the user mentions doutrina, PlayBook,
  ou revisao de sessao.
---

# Revisao de sessao / risco Aether

Ler `docs/llm-trading-doctrine.md` antes de concluir. LLM nao decide trade; avalia processo.

## Quando usar

- Usuario cola logs CLUSTER / EXEC / KELLY / RISK
- PR ou diff em `execution_*`, `domain/risk`, `sample_size_policy`, `settings.json` de risco
- Pedido de pos-mortem ou “por que perdemos”

## Pre-trade (PlayBook)

1. Qual setup nomeado? (ex.: explore TCN Cal>=floor + edge>=0)
2. Qual veto explicito? (cal_margin_floor, meta_negative_edge, adverse_path, SIDE_EQ, …)
3. Explore ou recover? Ha pending/linear?
4. Hipotese falsificavel da mudanca de knob (se houver)?

## During (leitura de log)

Ordem obrigatoria:

1. CLUSTER — Prob / Cal / Margin / Edge
2. SIDE_EQ / META_VETO — bias, N, soft/hard
3. IND — contexto TA, nao oraculo
4. KELLY — mode, live_n, f*
5. EXEC / EMPTY / PAUSE — `gate_reason`
6. RESOLVED / RISK — pending, linear, pnl_sess

Knobs de telemetria: `logging.level` (INFO/DEBUG), `logging.quiet_channels` (SETTLE/WARMUP/flow em DEBUG).
Prefixo opcional `[cN|SYM]` correlaciona o ciclo; tags SETTLE usam `SETTLE.{canal}:`.

Marcar cada ciclo como: **processo ok** | **processo falhou** | **inconclusivo (N baixo)**.

## Pos-mortem (9 perguntas)

1. Taleb: confundimos streak com edge?
2. Mlodinow: N suficiente para a conclusao?
3. Ellenberg: taxa-base / ACC / Bayes respeitados?
4. Duke: julgamos processo ou so P&L?
5. Bernstein: risco estava limitado (caps, pending)?
6. Douglas: houve revenge sizing ou troca de regras mid-session?
7. PlayBook: setup e veto estavam escritos? (ver tambem `aether-binary-senior`)
8. Murphy: TA substituiu a TCN ou so filtrou?
9. LTCM: algum fail-safe foi removido ou bypassado?

## Saida esperada

Resposta curta em PT-BR:

- Veredito do processo (nao “sorte”)
- Ciclos problematicos com `gate_reason` / Cal / Edge
- Acoes: manter knobs | ajuste minimo nomeado | retreino se ACC estruturalmente baixo
- Nunca recomendar `force_trade_every_cycle=true` como correcao
