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

Knobs: `compounding_rate_daily` **0.0431**; `payout_estimate` / `default_payout` **0.85**; `stop_win_kelly_cycles_target` **1**; `stop_win_kelly_live_n_min` **12**; `max_stake_pct` **0.05**. Soft recovery: `cover_enabled` **true**, `cover_multiple` **1.0**, amort **1/1**; stake = `min(max(PEND/payout, 1% banca), cap_L0)`; `recovery_cap_mode=cover_l0` (L0 **3.5%**, ignora L2/L3); PEND material nao force-explore por near-stop; skips de sinal waived com PEND; piso Kelly **1%** soberano tambem em RECOVER residual. Sem revenge sizing. Cold start (`live_n < 12`): Kelly × `explore_stake_scale` (piso **0.40**), sem boost Single-Strike.

## Pre-trade (PlayBook)

1. Setup: TCN + FLIP + SKIP `neg_edge` (EXPLORE; waived com PEND) + Kelly/cover_l0 amort **1**
2. Bloqueio tecnico? (`training`/`data`/`deploy`/`predict_error` / stop-win / cooldown / pausa / WSS); sinal? (`SKIP:neg_edge`) — com PEND material `neg_edge` nao trava recover (skips de vela/scale/doji desativados)
3. Explore ou recover? Cover amort **1** → `min(max(PEND/payout, 1% banca), cap_L0)`; telemetria `cover_l0`
4. Hipotese falsificavel se mudar knob; gate novo so via catalogo
5. Alvo: stop-win **4,31%** — processo, nao “mao quente”
6. Execucao: TCN (Cal ≥ 0.5 CALL) ou FLIP loss-clf (se pe no piso)

## During (leitura de log) — ordem CLUSTER→GATES→KELLY→EXEC→RESOLVED

1. CLUSTER — lado TCN (Cal≥0.5 CALL); Prob / Margin / Edge e `be`; conferir o payout liquido cotado no `[EXEC]` (`RATE`/`BE`) contra o usado no ciclo seguinte; Edge ≤ 0 → `SKIP:neg_edge` em EXPLORE (waived somente com PEND)
2. GATES — `[GATES] || LOSS_CLF` FLIP|OK|off; `blocked=bootstrap` / `flip_min_n` = FLIP off; se FLIP, lado = !TCN (sem bloqueio por vela)
3. KELLY — p, live_wr, f*, mode; sem meta_soft; stake = explore/recover/cover
4. EXEC — lado final: TCN ou FLIP; ticket em uma linha
5. RESOLVED / RISK — `LIN:` pos-settle; pending; `QUALITY hit=` so pos-FLIP; pnl vs 4.31%; ACC ~0.53 = retreino TCN, nao “mais trades”

## Pos-mortem (9 perguntas)

1. Taleb: confundimos streak com edge?
2. Mlodinow: `live_n` suficiente?
3. Ellenberg: taxa-base / ACC / Bayes?
4. Duke: processo ou so P&L curto?
5. Bernstein: caps / cover off / piso 1%?
6. Douglas: revenge sizing?
7. PlayBook: setup e bloqueio escritos? FLIP vs tecnico?
8. Murphy: TA substituiu TCN ou so telemetria?
9. LTCM: fail-safe (deploy/ACC/caps/`hard_p_loss_floor`) removido?

## Saida esperada

- Veredito do processo
- Ciclos com Cal / Edge / LOSS_CLF FLIP|OK
- Acoes: manter | ajuste minimo nomeado | retreino se ACC estruturalmente baixo
- Nunca `force_trade_every_cycle`; nunca rearmar quality gate amplo; nunca Soft do loss-clf
