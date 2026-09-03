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

Knobs: `compounding_rate_daily` **0.0431**; `payout_estimate` / `default_payout` **0.85**; `stop_win_kelly_cycles_target` **1**; `stop_win_kelly_min/max_fraction` **1.0**; `max_stake_pct` **0.05**; `stop_win_kelly_min_conviction` **0.52**. Soft discord: `max_stake_pct_discord` **0.05** / `kelly_mult_discord` **0.55** / `soft_max_stake_pct_high` **0.05** (nao capar Single-Strike em 1%). Meta atingida → `STOP_WIN` / `EXEC_PAUSE`. Soft recovery: `cover_multiple` **1.10**, amort **2/3**, `max_safe_stake_pct` **0.035**. Sem revenge sizing.

## Pre-trade (PlayBook)

1. Qual setup nomeado? (ex.: TCN resolve lado + fusao EV + Kelly; anti-loss microestrutura M5)
2. Qual bloqueio tecnico explicito? (`training`/`data`/`deploy`/`predict_error` / `neutral_zone` / stop-win)
3. Explore ou recover? Ha pending/linear? Recovery = amortização equilibrada em 2 a 3 ciclos (`cover_multiple` **1.10**); RECOVER stake com cap seguro (`max_safe_stake_pct: 0.035`).
4. Hipotese falsificavel da mudanca de knob (se houver)?
5. Alvo de negocio: stop-win **4,31%** composto — tacada única M5 com payout ~85%, progresso de processo, nao “mao quente”.

## During (leitura de log)

Ordem obrigatoria:

1. CLUSTER — Prob / Cal / Margin / Edge (telemetria); TF micro **M5** (300 s); anotar `live_n`
2. SCALE — MACRO (D1)/MICRO (M5)/MINI (M5)/MILI (ticks) + `tape`/`adapted` (adaptacao sob raw_extreme; soft Kelly; sem SKIP por escala)
3. GATES — `[GATES] || FUSION` (`ev_c`/`ev_p`/`why`) **antes** de loss-clf; `fusion_block_when_tcn_candle_agree` **false** (fusao livre mesmo se janela ops==TCN); depois `LOSS_CLF` SOFT vs FLIP (ref TCN); `FLIP_BLOCK:seed_candle|tcn_edge|seed|scale`; depois `[GATES] || ANTI_LOSS` com ancora hibrida + microestrutura M5 (EMA slope 2-pontos, RSI 0.35/0.65); por fim NEG_EDGE (soft se Edge<=0 sob `neg_edge_hard_skip=false`; hard Z-score panic com `Z=`/`side`/`thr`). Caveat: `fusion_loss_weight` nao ve `p_loss` do mesmo ciclo (FLIP apos fusao); seed `loss_bonus=0`. Se `auto=0` e FUSION != TCN com janela==TCN → regressao (seed nao pode puxar lado via loss_bonus). **M5 last-bar = log; confirmacao = janela N=3 / ancora hibrida.**
4. EXEC / EMPTY / PAUSE / COOLDOWN — `gate_reason` tecnico (`anti_loss_ema_slope`, `anti_loss_rsi_momentum`, `live_exec_discord`, `neg_edge_zscore_panic` com `Z=`/`side`/`thr`, `neutral_zone`) ou `signal_skip` 1.1; SIDE_EQ / scale = soft sizing; stop-win = `EXEC_PAUSE`; cooldown pós-loss = 1 ciclo (300s) se $L_2+$.
5. RESOLVED / RISK — pending, linear, pnl_sess vs alvo **4,31%**; polling de 2.0s em caso de estagnação de liquidação.

Marcar cada ciclo como: **processo ok** | **processo falhou** | **inconclusivo (N baixo)**.

Notas: `raw_extreme` mantem Cal para Kelly (nao e override MACRO TF). Escopo **1.1**: catalogo `signal_skip` fechado; quality gate amplo fora. WR bruto ≠ edge — P&L pode falhar com WR “bom” se lado/sizing errados. Soft (`fusion_weak_ev` / `neg_edge_soft`) continua EXEC: com SSOT `neg_edge_hard_skip` **false**, Edge `<= 0` ou subfloor → soft EXEC (processo coerente; **nao** e fusao lavando Cal). Hard `neg_edge` so com override `neg_edge_hard_skip` **true**. Seletividade de lado vem dos guards TCN/vela/auto_learn. Regressao de fusao so se hard_skip **true** e Edge Cal `<= 0` ainda EXEC. Se `FUSION p_eff` alto e Kelly `p≈be` → regressao (`kelly_used_fusion_p_eff`). Se `[RESOLVED] PEND` nao cai apos WIN → regressao de audit. Se `f*≈0` / `p≈be` / Margin~0 e stake ≈ primeira Kelly boa da sessao (`dlambert_unit` sticky) sob `neg_edge_soft` / near_stop / `recovery_force_explore` → **regressao U sticky** (EXPLORE forçado = piso `neutral_bankroll_pct`; cover so no DAL). Se soft + PEND material + stake ≈ CAP linear3 → **regressao cover-pleno** (revenge via cover sob soft).

## Pos-mortem (9 perguntas)

1. Taleb: confundimos streak com edge? WR de sessao vs Cal/Edge medio?
2. Mlodinow: `live_n` suficiente para a conclusao?
3. Ellenberg: taxa-base / ACC / Bayes respeitados? Comparar taxa-base do **lado no treino** (`label_call_frac`) vs distribuicao live CALL/PUT — vies de treino ≠ motivo para rearmar quality gate.
4. Duke: julgamos processo ou so P&L de poucos ciclos?
5. Bernstein: risco estava limitado (caps, pending, amort **2/3**, cover **1.10**)? SIDE_EQ / scale soft Kelly vs SKIP indevido?
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
