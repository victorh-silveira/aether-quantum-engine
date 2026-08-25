---
name: aether-binary-senior
description: >-
  Avalia sessoes live no estilo trader senior de opcoes binarias M15
  (CALL/PUT/SKIP tecnico ou signal_skip 1.1; OHLC 900s) no indice
  stp_500 (S&P 500). Use when analyzing CLUSTER/Cal/Edge logs, gate_reason
  tecnico, or when the user mentions playbook senior, SKIP, S&P 500, M15, ou binarias stp_500.
---

# Playbook senior binario (`stp_500` / M15)

Ler `docs/binary-senior-playbook.md` e `docs/deriv-indices-algorithm.md`.

Universo: **S&P 500** (`stp_500`) — **M15** (contrato ops **15 m / M15**; label TCN **N=1 vela M15**; ciclo **900 s**; micro/MINI **900 s**; macro **86400 s**; ratio **1:96**).

## Checklist (escopo 1.1)

1. Bloqueio tecnico? (`training`/`data`/`deploy`/`predict_error` / stop-win `EXEC_PAUSE`) — senao segue TCN/SCALE
2. Catalogo `signal_skip`? mini/cal = soft Kelly — senao candidato segue (sem flip pos-LOSS)
3. ACC/deploy de treino >= 0.53 quando o tema for modelo; checar `label_call_frac` / majority-collapse
4. Cal/Margin/Edge — telemetria; ticket so se Edge Cal TCN (`CLUSTER`) > 0; Kelly usa Cal (sem fusao) ou `fusion_p_eff` do lado escolhido **depois** do gate `neg_edge`; `live_n` para julgar WR vs P&L
5. SCALE dirs + **fusao EV** (`fusion_ev_call`/`fusion_ev_put`)? preferir argmax EV; `fusion_block_when_tcn_pos_edge` **true** preserva TCN so se Cal **e** raw +EV; `fusion_block_when_tcn_candle_agree` **true** preserva TCN se **janela ops N=3**==TCN (`why=tcn_candle_agree`); `fusion_loss_requires_auto_learn` **true** — seed nao alimenta loss_bonus; `fusion_loss_weight` **nao** ve `p_loss` do mesmo ciclo (FLIP apos fusao); EV fraco → soft Kelly `fusion_weak_ev_soft_kelly_mult` **0.40**; sob seed e ambos EV < 0 → `fusion_weak_ev_seed_soft_kelly_mult` **0.25**; log `[GATES] || FUSION`. **M15 last-bar = log; confirmacao = janela N=3.**
6. Loss-clf FLIP **ultimo** (ref TCN): soft vs FLIP (`p_loss>=0.90` + `veto_ready` + `flip_require_auto_learn` **true** — seed so SOFT; waivers `p_ovr`/`seed_discord` **nao** FLIP com `auto=0`); **nunca FLIP** se Edge Cal **e** raw >= floor **e** tape/janela ops confirmam TCN (`FLIP_BLOCK:tcn_edge`; tape ou janela ≠ TCN → `flip_waive_tcn_pos_edge_on_discord`); Cal+/raw− libera; sob seed bloquear contra janela (`FLIP_BLOCK:seed_candle`; `p_ovr` nao fura); `flip_seed_waive_edge_min` **−0.08**; live `flip_waive_edge_min` **-1.0**; janela no alvo floor **0.85** so se TCN fraco; primeiro fit apos `n>=READY` (**24**); `collapsed` zera bonus; esperar `auto=1` apos settles mistos
7. Chop = soft Kelly; **neg_edge** le Cal TCN (nao `fusion_p_eff`); soft se `0 < Cal < min_edge_execute` (`NEG_SOFT`; seed mult **0.25**); **hard** se Cal `<= 0` (`NEG_EDGE nonpos`; fusao nao lava); **trava de pânico Z-score bilateral**: CALL vetado se $Z < -2.0$, PUT vetado se $Z > +2.0$ (`gate_reason=neg_edge_zscore_panic`); hard `boot_deep` se seed+Cal < **−0.12** ou override `neg_edge_hard_skip`
8. RECOVER vs EXPLORE; pending/cover pleno (`cover_multiple` **1.50**, amort **1/1**, linear3 teto **2.5%**); damping stop-win inicio **1.0** / perto-meta **0.50**; RECOVER stake = cover (`f*` so gate); EXPLORE forçado / soft / `f*≈0` → piso **0.25%** banca + `explore_stake_scale_floor` **0.40** (**nunca** cover sob soft); Kelly Single-Strike **1%** ($\text{Stake} \approx 1,18\%$ da banca com payout 85%). Cooldown técnico: 1 ciclo ($900\,\text{s}$) ativado após $L_2+$ (`consecutive_losses_linear >= 2`), com transição imediata ($0\,\text{s}$) em $L_1$.
9. **Anti-loss com microestrutura M15 balanceada**:
   - **EMA Slope & Trend M15**: CALL exige `Preco > EMA9` em 15m; PUT exige `Preco < EMA9`. Se a inclinação da média de 21 for contrária, emite veto imediato `why=anti_loss_ema_slope` e retorna `EXEC_EMPTY`.
   - **Zero Bypass de Vela M15**: Se intenção for CALL e candle M15 anterior for PUT (`close < open`), ou vice-versa, veto mandatório `live_exec_discord` sem tolerância por tamanho de corpo.
   - **RSI Momentum**: CALL vetado se $\text{RSI}_{\text{M15}} < 0.38$; PUT vetado se $\text{RSI}_{\text{M15}} > 0.62$ (`why=anti_loss_rsi_momentum`).
   - **Seed**: unstamped + `p_loss`≥**0.85** + TCN pos_edge → **hard SKIP** se a janela ops nao confirma TCN com corpo minimo **0.10**; **live**: tier fraco (corpo liquido < **0.10** ou ausente); `gate_reason=anti_loss_seed_discord`; log `[GATES] || ANTI_LOSS`.
10. EXEC_EMPTY tecnico = sucesso de processo quando coerente (preserva a banca em condições assimétricas desfavoráveis); soft sinal (mini/cal/chop / neg_edge subfloor) continua EXEC — EMPTY de sinal `neg_edge` se Cal TCN `<= 0`, trava de pânico Z-score ou `anti_loss_*`; soft continua EXEC mas tamanho ≠ revenge via U sticky

## Proibido

- Revenge sizing pos-LOSS; force_trade_every_cycle como fix de EXEC_EMPTY
- Quality gate amplo (RSI/price_zone/SIDE_EQ block) como “solucao”
- Narrar mao quente / reversao sem evidencia de log
- Reativar `invert_exec_side` sem mandato e evidencia

## Referencias

- `docs/llm-trading-doctrine.md`
- `docs/binary-senior-playbook.md`
- `AGENTS.md`
