---
name: aether-binary-senior
description: >-
  Avalia sessoes live e verifica a esteira M5 (anti-loss, fusao EV, EMA slope,
  RSI) no estilo trader senior de opcoes binarias (CALL/PUT/SKIP tecnico ou
  signal_skip 1.1; OHLC 300s) no indice 1HZ75V. Use when analyzing
  CLUSTER/Cal/Edge logs, gate_reason tecnico, microestrutura M5, or when the
  user mentions playbook senior, SKIP, V75, M5, ou binarias 1HZ75V.
---

# Playbook senior binario (`1HZ75V` / M5)

Ler `docs/binary-senior-playbook.md` e `docs/deriv-indices-algorithm.md`.

Universo: **Volatility 75 (1s) Index** (`1HZ75V`) — **M5** (contrato ops **5 m / M5**; label TCN **N=1 vela M5**; ciclo **120 s**; micro/MINI **300 s**; macro **86400 s**; ratio **1:288**).

## Pipeline de decisao M5

1. Ciclo sincronizado ao fecho M5 (`cycle_interval_seconds` **120**, `signature_boundary_seconds` **300**)
2. Fusao EV (argmax CALL/PUT) com pesos SSOT; soft EV fraco **0.50** / seed **0.25**
3. Anti-loss: ancora hibrida ops N=3 + ultima vela; EMA9 slope rapido (`slope_tol * 0.6`) + EMA21 2-pontos; RSI de `micro_indicators` (**0.35/0.65**); corpo liquido janela ops
4. Neg-edge + panico Z-score bilateral ($Z < -2$ CALL / $Z > +2$ PUT)

## Checklist (escopo 1.1)

1. Bloqueio tecnico? (`training`/`data`/`deploy`/`predict_error` / stop-win `EXEC_PAUSE`) — senao segue TCN/SCALE
2. Catalogo `signal_skip`? mini/cal = soft Kelly — senao candidato segue (sem flip pos-LOSS)
3. ACC/deploy de treino >= 0.53 quando o tema for modelo; checar `label_call_frac` / majority-collapse
4. Cal/Margin/Edge — telemetria; ticket so se Edge Cal TCN (`CLUSTER`) > 0; Kelly usa Cal (sem fusao) ou `fusion_p_eff` do lado escolhido **depois** do gate `neg_edge`; `live_n` para julgar WR vs P&L
5. SCALE dirs + **fusao EV** (`fusion_ev_call`/`fusion_ev_put`)? preferir argmax EV; `fusion_block_when_tcn_pos_edge` **true** preserva TCN so se Cal **e** raw +EV; `fusion_block_when_tcn_candle_agree` **false** (fusao livre quando janela ops==TCN); `fusion_loss_requires_auto_learn` **true** — seed nao alimenta loss_bonus; `fusion_loss_weight` **nao** ve `p_loss` do mesmo ciclo (FLIP apos fusao); EV fraco → soft Kelly `fusion_weak_ev_soft_kelly_mult` **0.50**; sob seed e ambos EV < 0 → `fusion_weak_ev_seed_soft_kelly_mult` **0.25**; log `[GATES] || FUSION`. **M5 last-bar = log; confirmacao = janela N=3.**
6. Loss-clf FLIP **ultimo** (ref TCN): soft vs FLIP (`p_loss>=0.90` + `veto_ready` + `flip_require_auto_learn` **true** — seed so SOFT; waivers `p_ovr`/`seed_discord` **nao** FLIP com `auto=0`); **nunca FLIP** se Edge Cal **e** raw >= floor **e** tape/janela ops confirmam TCN (`FLIP_BLOCK:tcn_edge`; tape ou janela ≠ TCN → `flip_waive_tcn_pos_edge_on_discord`); Cal+/raw− libera; sob seed bloquear contra janela (`FLIP_BLOCK:seed_candle`; `p_ovr` nao fura); `flip_seed_waive_edge_min` **−0.08**; live `flip_waive_edge_min` **-1.0**; janela no alvo floor **0.85** so se TCN fraco; primeiro fit apos `n>=READY` (**24**); `collapsed` zera bonus; esperar `auto=1` apos settles mistos
7. Chop = soft Kelly; **neg_edge** com `neg_edge_hard_skip` **false** (SSOT): soft Kelly + EXEC se Edge `<= 0` ou subfloor (`NEG_SOFT`; seed mult **0.25**); **hard** so com override `neg_edge_hard_skip` **true** (`NEG_EDGE nonpos` / `boot_deep` se seed+Cal < **−0.12**); **trava de pânico Z-score bilateral**: CALL vetado se $Z < -2.0$, PUT vetado se $Z > +2.0$ (`gate_reason=neg_edge_zscore_panic`; telemetria `Z=`/`side`/`thr`)
8. RECOVER vs EXPLORE; pending/cover pleno (`cover_multiple` **1.10**, amort **2/3**, linear3 teto **3.5%**); damping stop-win inicio **1.0** / perto-meta **0.50**; RECOVER stake = cover (`f*` so gate); EXPLORE forçado / soft / `f*≈0` → piso **0.25%** banca + `explore_stake_scale_floor` **0.40** (**nunca** cover sob soft); Kelly Single-Strike **4.31%** ($\text{Stake} \approx 5,0\%$ da banca com payout 85%). Cooldown técnico: 1 ciclo ($300\,\text{s}$) ativado após $L_2+$ (`consecutive_losses_linear >= 2`), com transição imediata ($0\,\text{s}$) em $L_1$.
9. **Anti-loss com microestrutura M5 balanceada**:
   - **Ancora hibrida**: janela ops N=3 velas M5 + ultima vela micro fechada; telemetria `anti_loss_anchor_mode=hybrid`.
   - **EMA Slope & Trend M5**: CALL exige `Preco > EMA9` em 5m; PUT exige `Preco < EMA9`. EMA9 slope 2-pontos + EMA21 slope 2-pontos; cache por ciclo (`invalidate_ema_cache`). Veto `why=anti_loss_ema_slope`.
   - **Zero Bypass de Vela M5**: Se intenção for CALL e candle M5 anterior for PUT (`close < open`), ou vice-versa, veto mandatório `live_exec_discord` sem tolerância por tamanho de corpo.
   - **RSI Momentum**: CALL vetado se $\text{RSI}_{\text{M5}} < 0.35$; PUT vetado se $\text{RSI}_{\text{M5}} > 0.65$ (`why=anti_loss_rsi_momentum`).
   - **Seed**: unstamped + `p_loss`≥**0.85** + TCN pos_edge → **hard SKIP** se a janela ops nao confirma TCN com corpo minimo **0.10**; **live**: tier fraco (corpo liquido < **0.10** ou ausente) + confirmacao EXEC com `anti_loss_live_confirm_min_body` **0.25**; `gate_reason=anti_loss_seed_discord`; log `[GATES] || ANTI_LOSS`.
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
