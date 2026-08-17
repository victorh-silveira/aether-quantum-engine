---
name: aether-binary-senior
description: >-
  Avalia sessoes live no estilo trader senior de opcoes binarias M1
  (CALL/PUT/SKIP tecnico ou signal_skip 1.1; OHLC 60s) no indice
  R_10 (Volatility 10). Use when analyzing CLUSTER/Cal/Edge logs, gate_reason
  tecnico, or when the user mentions playbook senior, SKIP, Volatility, M1, ou binarias R_10.
---

# Playbook senior binario (`R_10` / M1)

Ler `docs/binary-senior-playbook.md` e `docs/deriv-indices-algorithm.md`.

Universo: **Volatility 10** (`R_10`) — **M1** (contrato ops **5 m / M5**; label TCN **N** ∈ {15,20,…,60}, **SSOT atual N=55** / H55; ciclo **60 s**; micro/MINI **60 s**; macro **7200 s**; ratio **1:120**).

## Checklist (escopo 1.1)

1. Bloqueio tecnico? (`training`/`data`/`deploy`/`predict_error` / stop-win `EXEC_PAUSE`) — senao segue TCN/SCALE
2. Catalogo `signal_skip`? mini/cal = soft Kelly — senao candidato segue (sem flip pos-LOSS)
3. ACC/deploy de treino >= 0.53 quando o tema for modelo; checar `label_call_frac` / majority-collapse
4. Cal/Margin/Edge — telemetria; Kelly usa Cal (sem fusao) ou `fusion_p_eff` do lado escolhido (com `fusion_applied`); `live_n` para julgar WR vs P&L
5. SCALE dirs + **fusao EV** (`fusion_ev_call`/`fusion_ev_put`)? preferir argmax EV; `fusion_block_when_tcn_pos_edge` **true** preserva TCN so se Cal **e** raw +EV; `fusion_block_when_tcn_candle_agree` **true** preserva TCN se vela==TCN (`why=tcn_candle_agree`); `fusion_loss_requires_auto_learn` **true** — seed nao alimenta loss_bonus; `fusion_loss_weight` **nao** ve `p_loss` do mesmo ciclo (FLIP apos fusao); EV fraco → soft Kelly `fusion_weak_ev_soft_kelly_mult` **0.40**; sob seed e ambos EV &lt; 0 → `fusion_weak_ev_seed_soft_kelly_mult` **0.25**; log `[GATES] || FUSION`
6. Loss-clf FLIP **ultimo** (ref TCN): soft vs FLIP (`p_loss>=0.90` + `veto_ready` + `flip_require_auto_learn` **true** — seed so SOFT); **nunca FLIP** se Edge Cal **e** raw >= floor (`FLIP_BLOCK:tcn_edge`; Cal+/raw− libera); sob seed bloquear contra vela (`FLIP_BLOCK:seed_candle`; `p_ovr` nao fura); `flip_seed_waive_edge_min` **−0.08**; live `flip_waive_edge_min` **-1.0**; vela no alvo floor **0.85** so se TCN fraco; primeiro fit apos `n>=READY` (**24**); `collapsed` zera bonus; esperar `auto=1` apos settles mistos
7. Chop = soft Kelly; **neg_edge** soft se `0 < edge < min_edge_execute` (`NEG_SOFT`; seed mult **0.25**); **hard** se `edge <= 0` (`NEG_EDGE nonpos`); hard `boot_deep` se seed+edge &lt; **−0.12** ou override `neg_edge_hard_skip`
8. RECOVER vs EXPLORE; pending/cover pleno (`cover_multiple` **1.50**, amort **1/1**, linear3 teto **2.5%**); damping stop-win inicio **1.0** / perto-meta **0.50**; RECOVER stake = cover (`f*` so gate); EXPLORE forçado / soft / `f*≈0` → piso **0.25%** banca + `explore_stake_scale_floor` **0.40** (**nunca** cover sob soft); Kelly `fraction` **0.08**; caps stop-win Kelly ate **5%**. Se `f*≈0` e stake ≈ 1ª Kelly da sessao sob soft/freeze → regressao **U sticky**; se soft + PEND + stake ≈ CAP linear3 → regressao **cover-pleno**
9. **Anti-loss seed discord**: seed + `p_loss`≥**0.85** + vela≠TCN + TCN pos_edge → **hard SKIP** EXPLORE e RECOVER (`gate_reason=anti_loss_seed_discord`); vela do gate = vela do `[CANDLE]`, nao a anterior do snapshot; soft so se `anti_loss_hard_skip` **false** (`[GATES] || ANTI_LOSS`)
10. EXEC_EMPTY tecnico = sucesso de processo quando coerente; soft sinal (mini/cal/chop / neg_edge subfloor) continua EXEC — EMPTY de sinal `neg_edge` se `edge <= 0` (ou seed profundo / override) ou `anti_loss_seed_discord`; soft continua EXEC mas tamanho ≠ revenge via U sticky

## Proibido

- Revenge sizing pos-LOSS; force_trade_every_cycle como fix de EXEC_EMPTY
- Quality gate amplo (RSI/price_zone/SIDE_EQ block) como “solucao”
- Narrar mao quente / reversao sem evidencia de log
- Reativar `invert_exec_side` sem mandato e evidencia

## Referencias

- `docs/llm-trading-doctrine.md`
- `docs/binary-senior-playbook.md`
- `AGENTS.md`
