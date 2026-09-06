---
name: aether-binary-senior
description: >-
  Avalia sessoes live e verifica a esteira M5 (regime boolean, fusao EV,
  neg_edge) no estilo trader senior de opcoes binarias (CALL/PUT/SKIP tecnico ou
  signal_skip 1.1; OHLC 300s) no indice 1HZ75V. Use when analyzing
  CLUSTER/Cal/Edge logs, gate_reason tecnico, microestrutura M5, or when the
  user mentions playbook senior, SKIP, V75, M5, ou binarias 1HZ75V.
---

# Playbook senior binario (`1HZ75V` / M5)

Ler `docs/binary-senior-playbook.md` e `docs/deriv-indices-algorithm.md`.

Universo: **Volatility 75 (1s) Index** (`1HZ75V`) — **M5** (contrato ops **5 m / M5**; label TCN **N=1 vela M5**; ciclo **300 s** com `require_signature_boundary` **true**; micro/MINI **300 s**; macro **86400 s**; ratio **1:288**). TCN ortogonal **14D** / meta **23D**; limiares **0.53/0.47**; `ema_50` **50**.

## Pipeline de decisao M5

1. Ciclo sincronizado a abertura M5 (`cycle_interval_seconds` **300**, `signature_boundary_seconds` **300**, `require_signature_boundary` **true**)
2. TCN 14D + limiares **0.53/0.47** + fusao EV (argmax CALL/PUT); soft EV fraco **0.50** / seed **0.25**
3. Loss-clf (soft / FLIP sob SSOT `flip_*`)
4. **Micro protect**: vela M5 ≠ EXEC + corpo minimo → FOLLOW Soft_SIZE se Edge Cal da vela >= **0.015** (`micro_discord_follow_candle` **true**); senao HARD `micro_discord`; soft/FLIP_BLOCK + `p_loss`>=**0.90** + vela ≠ EXEC → HARD `chop_loss_risk` (vela alinhada nao HARD); Soft so EXEC se `confirm_score` >= **2** (vela/tape/mi/mili/ops definidos); senao HARD `soft_confirm_weak`
5. **Regime boolean** (`regime_gate_enabled` **true**): ADX fraco + BB squeeze → HARD `regime_squeeze` **sem** alterar CALL/PUT
6. Anti-loss direcional **off** no SSOT (`anti_loss_allow_candle_flip` **false**, live confirm/weak **false**, seed discord **false`) — codigo legado so para testes
7. Neg-edge + panico Z-score bilateral ($Z < -2$ CALL / $Z > +2$ PUT)

## Checklist (escopo 1.1)

1. Bloqueio tecnico? (`training`/`data`/`deploy`/`predict_error` / stop-win `EXEC_PAUSE` / `regime_squeeze`) — senao segue TCN/SCALE
2. Catalogo `signal_skip`? mini/cal = soft Kelly — senao candidato segue (sem flip pos-LOSS)
3. ACC/deploy de treino >= 0.53 quando o tema for modelo; checar `label_call_frac` / majority-collapse; ckpt **14D** fail-closed
4. Cal/Margin/Edge — telemetria; ticket so se Edge Cal TCN (`CLUSTER`) >= `min_edge_*` (**0.015**); Kelly usa Cal (sem fusao) ou `fusion_p_eff` do lado escolhido **depois** do gate `neg_edge`; `live_n` para julgar WR vs P&L
5. SCALE dirs + **fusao EV** (`fusion_ev_call`/`fusion_ev_put`)? preferir argmax EV; `fusion_block_when_tcn_pos_edge` **true** preserva TCN so se Cal **e** raw +EV; `fusion_block_when_tcn_candle_agree` **false**; `fusion_loss_requires_auto_learn` **true**; soft Kelly EV fraco **0.50** / seed dual-EV<0 **0.25**; log `[GATES] || FUSION`
6. Loss-clf FLIP **ultimo** (ref TCN): soft vs FLIP (`p_loss` >= floor; com `flip_waive_guards_above_p_loss` **0.85** fura seed/tcn_edge/seed_candle mesmo vela alinhada); pos-FLIP micro nao desfaz; Edge Cal fraco → Soft_SIZE via neg_edge mesmo se vela ≠ EXEC
7. Chop = soft Kelly; **neg_edge** com `neg_edge_hard_skip` **true**: Edge `<= 0` → HARD; Edge `< min_edge_*` (**0.015**) → HARD `neg_edge_subfloor_hard`; Soft_SIZE so soft flags **com** Edge >= floor; Soft EXEC so se `confirm_score` >= **2** (senao HARD `soft_confirm_weak`); **trava Z-score**: CALL se $Z < -2.0$, PUT se $Z > +2.0$
8. RECOVER vs EXPLORE; `cover_enabled` **false**; caps linear3 teto **3.5%**; damping stop-win inicio **1.0** / perto-meta **0.50**; stake = Kelly + piso **1%**; Soft_SIZE **2.5%** se Edge>=0.015 **tambem com PEND**; Kelly Single-Strike **4.31%** so em `gate_verdict=ALLOW`. Cooldown: 1 ciclo ($300\,\text{s}$) apos $L_2+$.
9. **Regime-only**: `REGIME || HARD_SKIP why=regime_squeeze` = processo correto em squeeze; nao “corrigir” com flip de vela
10. EXEC_EMPTY tecnico = sucesso quando Edge `<= 0` / Edge `< floor` / `neutral_zone` / `regime_squeeze` / `micro_discord` / `chop_loss_risk` / `soft_confirm_weak`

## Proibido

- Revenge sizing pos-LOSS; force_trade como fix de EXEC_EMPTY / Edge<=0 / Edge < floor / neutral_zone / regime_squeeze
- Julgar EMPTY Edge<=0 ou Edge < floor ou regime como bug de frequencia; reabrir `neg_edge_hard_skip`; Soft_SIZE sob Edge subfloor
- Soft_SIZE + Edge forte com stake &lt; 1%; Soft_SIZE+PEND+Edge>=0.015+stake≈1%; EXEC stake &lt; 1% banca; PEND + stake ≈ cover amort
- Quality gate amplo (RSI/price_zone/SIDE_EQ block) como “solucao”
- Reativar flip/live confirm/seed discord anti-loss sem mandato e evidencia
- Narrar mao quente / reversao sem evidencia de log
- Reativar `invert_exec_side` sem mandato e evidencia

## Referencias

- `docs/llm-trading-doctrine.md`
- `docs/binary-senior-playbook.md`
- `AGENTS.md`
