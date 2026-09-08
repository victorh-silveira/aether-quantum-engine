---
name: aether-session-review
description: >-
  Revisa sessoes live, logs do motor Aether, Kelly Single-Strike 4.31% e mudancas
  de risco/execucao com checklist PlayBook alinhada a doutrina LLM. Use when
  analyzing engine logs, EXEC_EMPTY, Kelly/recovery PRs, stop-win sizing,
  session post-mortems, or when the user mentions doutrina, PlayBook, Single-Strike,
  ou revisao de sessao.
---

# Revisao de sessao / risco Aether

Ler `docs/llm-trading-doctrine.md`, `docs/binary-senior-playbook.md` e `docs/engineering-indicator-gates.md` antes de concluir. LLM nao decide trade; avalia processo.

## Quando usar

- Usuario cola logs CLUSTER / SCALE / LOSS_CLF / EXEC / KELLY / RISK
- PR ou diff em `execution_*`, `domain/risk`, `sample_size_policy`, `settings.json` de risco
- Calibracao Single-Strike / stop-win **4.31%**
- Pedido de pos-mortem ou “por que perdemos”

## Kelly Single-Strike (SSOT)

$$\text{Lucro Alvo} = \text{Banca} \times 0.0431$$
$$\text{Stake} = \frac{\text{Banca} \times 0.0431}{0.85} \approx 0.0507 \times \text{Banca} \implies \text{cap } 5.0\%$$

Knobs: `compounding_rate_daily` **0.0431**; `payout_estimate` / `default_payout` **0.85**; `stop_win_kelly_cycles_target` **1**; `max_stake_pct` **0.05**. Soft recovery: `cover_enabled` **false**; piso Kelly **1%**; `max_safe_stake_pct` **0.035**. Sem revenge sizing.

## Pre-trade (PlayBook)

1. Setup: TCN resolve lado + HARD SKIP por P_LOSS + Kelly
2. Bloqueio tecnico? (`training`/`data`/`deploy`/`predict_error` / stop-win) ou `loss_clf`?
3. Explore ou recover? Com `cover_enabled` **false**, PEND nao infla stake
4. Hipotese falsificavel se mudar knob; gate novo so via catalogo
5. Alvo: stop-win **4,31%** — processo, nao “mao quente”

## During (leitura de log)

1. CLUSTER — Prob / Cal / Margin / Edge; `live_n`
2. SCALE — vision telemetria (sem adapt de lado)
3. GATES — `[GATES] || LOSS_CLF` HARD|OK|off; EMPTY tecnico ou `loss_clf` = processo ok quando coerente
4. KELLY / EXEC
5. RESOLVED / RISK — pending, linear, pnl vs 4.31%

## Pos-mortem (9 perguntas)

1. Taleb: confundimos streak com edge?
2. Mlodinow: `live_n` suficiente?
3. Ellenberg: taxa-base / ACC / Bayes?
4. Duke: processo ou so P&L curto?
5. Bernstein: caps / cover off / piso 1%?
6. Douglas: revenge sizing?
7. PlayBook: setup e bloqueio escritos? HARD `loss_clf` vs tecnico?
8. Murphy: TA substituiu TCN ou so telemetria?
9. LTCM: fail-safe (deploy/ACC/caps/`hard_p_loss_floor`) removido?

## Saida esperada

- Veredito do processo
- Ciclos com `gate_reason` / Cal / Edge / LOSS_CLF
- Acoes: manter | ajuste minimo nomeado | retreino se ACC estruturalmente baixo
- Nunca `force_trade_every_cycle`; nunca rearmar quality gate amplo; nunca reabrir FLIP/signal_skip
