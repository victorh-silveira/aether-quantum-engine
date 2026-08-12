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

1. Qual setup nomeado? (ex.: TCN resolve lado + fusao EV + Kelly; soft SIDE_EQ / scale_vision)
2. Qual bloqueio tecnico explicito? (`training`/`data`/`deploy`/`predict_error` / stop-win)
3. Explore ou recover? Ha pending/linear? Recovery = cover amortizado (**1.50**, amort **2–4**); damping inicio **1.0** / perto-meta **0.50**; RECOVER stake ≠ `bankroll×f*` (`f*` so gate). `scale_force_explore` ou `RECOVERY_INFEASIBLE`/`recovery_force_explore` bloquearam RECOVER/DAL? Pending material com `pending_waives_scale_explore` deve liberar soft cover.
4. Hipotese falsificavel da mudanca de knob (se houver)?
5. Alvo de negocio: stop-win **3%** composto — progresso de processo, nao “mao quente”

## During (leitura de log)

Ordem obrigatoria:

1. CLUSTER — Prob / Cal / Margin / Edge (telemetria); TF tipicamente micro **M2** (120 s); anotar `live_n`
2. SCALE — MACRO/MICRO/MINI/MILI + `tape`/`adapted` (adaptacao sob raw_extreme; soft Kelly; sem SKIP por escala)
3. GATES — `[GATES] || FUSION` (`ev_c`/`ev_p`/`why`); `why=tcn_candle_agree` = TCN==vela (switch bloqueado); `LOSS_CLF` SOFT vs FLIP; `FLIP_BLOCK:seed_candle|tcn_edge|seed|scale`; NEG_EDGE soft. Se `auto=0` e FUSION != TCN com vela==TCN → regressao (seed nao pode puxar lado via loss_bonus).
4. EXEC / EMPTY / PAUSE — `gate_reason` tecnico ou `signal_skip` 1.1; SIDE_EQ / scale = soft sizing; stop-win = `EXEC_PAUSE`
5. RESOLVED / RISK — pending, linear, pnl_sess vs alvo **3%**

Marcar cada ciclo como: **processo ok** | **processo falhou** | **inconclusivo (N baixo)**.

Notas: `raw_extreme` mantem Cal para Kelly (nao e override MACRO TF). Escopo **1.1**: catalogo `signal_skip` fechado; quality gate amplo fora. WR bruto ≠ edge — P&L pode falhar com WR “bom” se lado/sizing errados. Soft (`fusion_weak_ev` / `neg_edge_soft`) continua EXEC — seletividade de lado vem dos guards TCN/vela/auto_learn, nao de SKIP amplo. Se `FUSION p_eff` alto e Kelly `p≈be` → regressao (`kelly_used_fusion_p_eff`). Se `[RESOLVED] PEND` nao cai apos WIN → regressao de audit. Se `f*≈0` / `p≈be` / Margin~0 e stake ≈ primeira Kelly boa da sessao (`dlambert_unit` sticky) sob `neg_edge_soft` / near_stop / `recovery_force_explore` → **regressao U sticky** (tamanho deve ser piso `neutral_bankroll_pct` ou cover∩piso, nao revenge via U).

## Pos-mortem (9 perguntas)

1. Taleb: confundimos streak com edge? WR de sessao vs Cal/Edge medio?
2. Mlodinow: `live_n` suficiente para a conclusao?
3. Ellenberg: taxa-base / ACC / Bayes respeitados? Comparar taxa-base do **lado no treino** (`label_call_frac`) vs distribuicao live CALL/PUT — vies de treino ≠ motivo para rearmar quality gate.
4. Duke: julgamos processo ou so P&L de poucos ciclos?
5. Bernstein: risco estava limitado (caps, pending, amort **2–4**, cover **1.50**)? SIDE_EQ / scale soft Kelly vs SKIP indevido?
6. Douglas: houve revenge sizing ou troca de regras mid-session?
7. PlayBook: setup e bloqueio tecnico estavam escritos? Houve chuva de FLIP seed (`auto=0`) vs `FLIP_BLOCK`?
8. Murphy: TA substituiu a TCN ou so filtrou telemetria (SCALE/fusao)?
9. LTCM: algum fail-safe tecnico (deploy/ACC/caps/`flip_require_auto_learn`) foi removido?

## Saida esperada

Resposta curta em PT-BR:

- Veredito do processo (nao “sorte”)
- Ciclos problematicos com `gate_reason` / Cal / Edge / FUSION / `FLIP_BLOCK` / SCALE discord
- Acoes: manter knobs | ajuste minimo nomeado | retreino se ACC estruturalmente baixo
- Nunca recomendar `force_trade_every_cycle=true` como correcao; nunca rearmar quality gate amplo
