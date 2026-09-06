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

Ler `docs/llm-trading-doctrine.md` e `docs/medallion.md` antes de concluir. LLM nao decide trade; avalia processo.

## Quando usar

- Usuario cola logs CLUSTER / SCALE / EXEC / KELLY / RISK
- PR ou diff em `execution_*`, `domain/risk`, `sample_size_policy`, `settings.json` de risco
- Calibracao Single-Strike / stop-win **4.31%**
- Pedido de pos-mortem ou “por que perdemos”

## Kelly Single-Strike (SSOT)

$$\text{Lucro Alvo} = \text{Banca} \times 0.0431$$
$$\text{Stake} = \frac{\text{Banca} \times 0.0431}{0.85} \approx 0.0507 \times \text{Banca} \implies \text{cap } 5.0\%$$

Knobs: `compounding_rate_daily` **0.0431**; `payout_estimate` / `default_payout` **0.85**; `stop_win_kelly_cycles_target` **1**; `stop_win_kelly_min/max_fraction` **1.0**; `max_stake_pct` **0.05**; `stop_win_kelly_min_conviction` **0.52**. Soft discord: `max_stake_pct_discord` **0.05** / `kelly_mult_discord` **0.55** / `soft_max_stake_pct_high` **0.05**. Meta atingida → `STOP_WIN` / `EXEC_PAUSE`. Soft recovery: `cover_enabled` **false**; piso Kelly **1%**; `max_safe_stake_pct` **0.035**. Sem revenge sizing.

## Pre-trade (PlayBook)

1. Qual setup nomeado? (ex.: TCN resolve lado + fusao EV + Kelly; anti-loss microestrutura M5)
2. Qual bloqueio tecnico explicito? (`training`/`data`/`deploy`/`predict_error` / `neutral_zone` / stop-win)
3. Explore ou recover? Ha pending/linear? Com `cover_enabled` **false**, PEND nao infla stake (Kelly + piso **1%** + caps `max_safe_stake_pct: 0.035`).
4. Hipotese falsificavel da mudanca de knob (se houver)?
5. Alvo de negocio: stop-win **4,31%** composto — tacada única M5 com payout ~85%, progresso de processo, nao “mao quente”.

## During (leitura de log)

Ordem obrigatoria:

1. CLUSTER — Prob / Cal / Margin / Edge (telemetria); TF micro **M5** (300 s); anotar `live_n`
2. SCALE — MACRO (D1)/MICRO (M5)/MINI (M5)/MILI (ticks) + `tape`/`adapted` (adaptacao sob raw_extreme; soft Kelly; sem SKIP por escala)
3. GATES — FUSION → LOSS_CLF → ANTI → MICRO → REGIME (`regime_squeeze` HARD sem flip; `REGIME_CHOP` soft) → NEG_EDGE (Edge<=0 HARD; Edge < **0.015** HARD `neg_edge_subfloor_hard`; Soft_SIZE so soft flags com Edge >= floor; Soft_SIZE piso 2.5% so se Edge>=0.015 tambem com PEND). EMPTY Edge<=0 / Edge&lt;floor / `neutral_zone` / `regime_squeeze` / `micro_discord` / `chop_loss_risk` / `soft_confirm_weak` = processo ok. Anti-loss direcional **off** no SSOT — nao narrar flip/RSI soft como filtro vivo.
4. EXEC / EMPTY — `regime_squeeze` / Edge<=0 / Edge&lt;floor / `neutral_zone` = HARD EMPTY (sucesso de processo quando coerente).
5. RESOLVED / RISK — pending, linear, pnl vs 4.31%.

Notas: Soft_SIZE sem Single-Strike; piso **2.5%** so com Edge >= **0.015**; Soft_SIZE so com Edge >= floor; D-SQUEEZE nao pode reduzir Soft_SIZE+Edge>=floor a `$1`. Nao julgar EMPTY Edge<=0 ou Edge&lt;floor ou regime como bug. Nao reabrir `neg_edge_hard_skip`.

## Pos-mortem (9 perguntas)

1. Taleb: confundimos streak com edge? WR de sessao vs Cal/Edge medio?
2. Mlodinow: `live_n` suficiente para a conclusao?
3. Ellenberg: taxa-base / ACC / Bayes respeitados? Comparar taxa-base do **lado no treino** (`label_call_frac`) vs distribuicao live CALL/PUT — vies de treino ≠ motivo para rearmar quality gate.
4. Duke: julgamos processo ou so P&L de poucos ciclos?
5. Bernstein: risco estava limitado (caps, pending, `cover_enabled` false, piso 1%)? SIDE_EQ / scale soft Kelly vs SKIP indevido?
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

- TCN ortogonal **14D** / meta **23D**; limiares **0.53/0.47**; `ema_50` **50**; anti-loss direcional off; `regime_gate_enabled` HARD squeeze.
