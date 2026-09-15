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

Knobs: `compounding_rate_daily` **0.0431**; `payout_estimate` / `default_payout` **0.85**; `stop_win_kelly_cycles_target` **1**; `stop_win_kelly_live_n_min` **12**; `max_stake_pct` **0.05**. Soft recovery: `cover_enabled` **true**, `cover_multiple` **1.0**; piso Kelly **1%**; `max_safe_stake_pct` **0.035**. Sem revenge sizing. Cold start (`live_n < 12`): Kelly × `explore_stake_scale` (piso **0.40**), sem boost Single-Strike.

## Pre-trade (PlayBook)

1. Setup: TCN resolve lado + FLIP por p_eff (auto_learn; young/mature pe>=**0.58**; `n_train>=8`; **nao** se vela==TCN) + SCALE adapt + Kelly
2. Bloqueio tecnico? (`training`/`data`/`deploy`/`predict_error` / stop-win / cooldown pos-LOSS / pausa de sessao / WSS down)
3. Explore ou recover? Com `cover_enabled` **true**, PEND material usa cover capped (`min(PEND/payout/amort, cap_L)`)
4. Hipotese falsificavel se mudar knob; gate novo so via catalogo
5. Alvo: stop-win **4,31%** — processo, nao “mao quente”

## During (leitura de log) — ordem CLUSTER→CANDLE→GATES→SCALE→META→KELLY

1. CLUSTER — lado TCN (Cal≥0.5 CALL); Prob / Margin / Edge (EV vs be=0.541); `live_n`; Edge negativo sozinho nao e SKIP
2. CANDLE — `dir_vela` same-cycle (o→c da M5 fechada); compara com TCN
3. GATES — `[GATES] || LOSS_CLF` FLIP|OK|off; `blocked=bootstrap` / `flip_min_n` / **`candle_holds`** = FLIP off; se FLIP, lado = !TCN so se vela ≠ TCN
4. IND SCALE — `adapted=` + `why=retract_vs_tcn|explos_vs_tcn|tape_vs_tcn|candle_vs_tcn|flip_holds|explos_edge_firm`; mi/mili/tape; regime falha → tape_strong; candle last (exceto FLIP sticky)
5. META — `applied=` / edge ≤ 0 ou `sat=1`+Cal Edge≤0.02 → soft Kelly (`meta_soft_kelly`; **nao flipa**); `edge=+0.850 sat=1` = clip
6. KELLY / EXEC — lado final deve bater a cascata; stake = explore/recover/cover
7. RESOLVED / RISK — `LIN:` pos-settle; pending; `QUALITY hit=` so pos-FLIP; pnl vs 4.31%; ACC ~0.53 = retreino TCN, nao “mais trades”

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
