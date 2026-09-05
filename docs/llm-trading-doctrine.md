# Doutrina LLM do Aether (9 livros)

O LLM/Cursor e **copiloto de engenharia e auditoria**. Nao decide CALL/PUT em runtime. A decisao live permanece TCN + meta LightGBM + Kelly/caps + **fusao EV multi-escala** (`execution_direction_fusion`); **escopo 1.1** + arquitetura continua 1HZ75V: catálogo `signal_skip` (`mini_pair_oppose` / `cal_margin` / chop = soft Kelly **0.55**); **neg_edge** le Edge do lado (`CLUSTER`); contrato `gate_verdict` **HARD_SKIP | SOFT_SIZE | ALLOW**: com `neg_edge_hard_skip` **true** (SSOT) Edge `<= 0` → HARD SKIP; Edge `< min_edge_*` (**0.015** explore = recovery, inclusive subfloor positivo) → HARD SKIP (`neg_edge_subfloor_hard`); Soft_SIZE so de soft flags **com** Edge >= floor (**sem** Single-Strike); Edge >= floor + sem soft → ALLOW (Single-Strike ok); trava Z-score panic permanece hard (`SKIP:NEG_EDGE_ZSCORE_PANIC` com telemetria `Z=`/`side`/`thr`); **Anti-loss** com microestrutura M5 (ancora hibrida ops N=3 + ultima vela fechada; `anti_loss_anchor_agree=false` + last-bar ≠ EXEC → soft `live_discord_weak` ou flip se Edge last >= floor; EMA slope 2-pontos 9/21; RSI **0.30/0.70**; confirm corpo ≥ **0.15**; `anti_loss_live_exec_candle_enabled` **false**; cache EMA por ciclo). `invert_exec_side` **false**; `online_training` **false**; stop-win **4.31%** Kelly Single-Strike.

SSOT operacional: [`config/settings.json`](../config/settings.json) — universo **`1HZ75V`** (Volatility 75 (1s) Index), contrato ops **5 m (M5)**, label TCN **N=1** vela M5 (`quantum_multi_barrier`), micro/MINI **300 s** (M5), ciclo **300 s** (`require_signature_boundary` **true**, abertura M5), macro **86400 s** (D1). Metodologia: [`medallion.md`](medallion.md). Sample size: [`sample-size-lln.md`](sample-size-lln.md). Loss ML: [`infra-docker.md`](infra-docker.md) (`aether-loss-classifier`). Arquitetura senior (host 3.13 / DDD / sidecars): [`engineering-architecture-senior.md`](engineering-architecture-senior.md).

---

## 1. Iludido pelo Acaso (Taleb)

**Insight:** streak e path-dependence nao sao edge; o acaso recompensa narrativas.

**Anti-padrao do LLM:** afrouxar piso apos 2–3 WINs; tratar Cal 0.51 como sinal; aumentar stake porque “estava batendo”.

**Regra no Aether:** caps limitam cauda; Cal/Edge sao telemetria; margem fraca via `signal_skip.cal_margin` (floor SSOT **0.022** → soft Kelly, waive pending) — nao quality gate generico.

**Ancoras:** soft recovery caps (`max_safe_stake_pct` **0.035**; linear2 **0.03**; linear3 **0.025**); Kelly; `force_trade_every_cycle=false`; cover inviavel (`RECOVERY_INFEASIBLE`) com PEND material → stake = **CAP** (cover parcial, DAL); `pending_waives_scale_explore` (pending material libera soft cover sob discord/adapt).

---

## 2. A Caminhada do Bebado (Mlodinow)

**Insight:** N pequeno e ruido; a media so converge com volume.

**Anti-padrao do LLM:** hard-skip ou flip apos 2 losses; misturar `live_wr` com N=1; “aprendemos rapido” via mais trades frios.

**Regra no Aether:** `sample_size_policy` (telemetria/sizing); explore stake scale com N baixo.

**Ancoras:** `sample_size_policy` (`evidence_n_min=12`, `large_n_min=32`, `explore_stake_scale_floor=0.40`); `app/src/domain/analytics/sample_size_policy.py`; [`sample-size-lln.md`](sample-size-lln.md).

---

## 3. Como Nao Errar (Ellenberg)

**Insight:** esperanca, Bayes e taxa-base; mudanca de knob precisa de hipotese falsificavel.

**Anti-padrao do LLM:** alterar threshold “porque o log ficou feio”; ignorar ACC de validacao; misturar prior e evidencia sem shrink; tratar streak de PUT LOSS como motivo para **rearmar quality gate** amplo (RSI/price_zone).

**Regra no Aether:** Kelly bayesiano; gate de `val_accuracy`; calib drift so com N minimo. Vies de lado: treino + SIDE_EQ soft Kelly. Catalogo sinal (`signal_skip`) = soft Kelly para mini/cal/chop; loss-clf = soft em `[veto_p_loss_floor, hard_p_loss_floor)` com `veto_ready` e **FLIP** CALL↔PUT **relativo ao TCN** se `p_loss >= hard_p_loss_floor` (**0.90**, `veto_ready`); seed bloqueia FLIP contra vela; chop = soft Kelly continuo; **neg_edge** com `neg_edge_hard_skip` **true** (SSOT): Edge `<= 0` → HARD (`gate_reason=neg_edge`); Edge `< min_edge_*` (**0.015** explore = recovery, inclusive `0 < Edge < floor`) → HARD (`neg_edge_subfloor_hard`); Soft_SIZE so de soft flags **com** Edge >= floor (sem Single-Strike). Override `neg_edge_hard_skip` **false** reabre soft em Edge `<= 0` / subfloor. Trava Z-score panic permanece hard. Kelly usa `fusion_p_eff` so apos o gate.

**Ancoras:** `risk_management.min_validation_accuracy_gate` (0.53); `soft_min_val_accuracy`; `deep_learning.sample_weighting`; `deploy_gate.reject_majority_collapse`; `execution_side_eq_sizing`; `apply_live_calib_drift_soft`; Kelly em `app/src/domain/risk/`.

---

## 4. Pensando em Apostas (Duke)

**Insight:** qualidade do processo > resultado do ciclo.

**Anti-padrao do LLM:** `force_trade_every_cycle=true` como “fix” de EXEC_EMPTY; julgar sessao so pelo P&L de 5 trades.

**Regra no Aether:** `EXEC_EMPTY` com `gate_reason` coerente (Edge `<= 0`, `neutral_zone`, seed unstamped) e sucesso de processo; force trade permanece off. Live confirm/discord/weak/RSI = soft (nao EMPTY). Nao julgar EMPTY Edge<=0 como bug de frequencia.

**Ancoras:** `force_trade_every_cycle: false`; logs `gate_reason` / `quality_gate_reason`; `execution_quality_reject.py`.

---

## 5. Desafio aos Deuses (Bernstein)

**Insight:** risco e o que se mede e limita; incerteza sem medida nao escala exposicao.

**Anti-padrao do LLM:** reinventar stop-loss interno; recovery sem pending; sizing sem teto de banca.

**Regra no Aether:** `cover_enabled` **false** — PEND material **nao** dimensiona stake por cover `pending/payout/amort`; stake = Kelly/EXPLORE + piso **1%** (`neutral_bankroll_pct` / `min_stake_pct` **0.01**) + caps L0/L1 **3.5%**; ledger WIN abate lucro real (sem zerar pending em massa via sizing). Soft_SIZE (so com Edge >= floor) eleva a **2.5%** (`soft_size_min_edge` **0.015**); Edge `< floor` e HARD via neg_edge (nao Soft_SIZE). Stop-win ativo; damping `target_damping_*` inicio **1.0** / perto-meta **0.50**; stop-loss interno desativado. **Nunca** amortizacao em massa / cover pleno; **nunca** `dlambert_unit` sticky.

**Ancoras:** `soft_recovery_policy`; `soft_recovery_explore.py`; `pending_loss`; stop-win composto; `stake_target_proximity.py`; `app/src/domain/risk/risk_recovery_state.py`.

---

## 6. Trading in the Zone (Douglas)

**Insight:** probabilismo; desapego ao ultimo trade; regras consistentes.

**Anti-padrao do LLM:** revenge sizing apos LOSS; mudar playbook no meio da sessao por emocao do log.

**Regra no Aether:** cooldown linear; RECOVER so com pending/linear; EXPLORE vs RECOVER distintos.

**Ancoras:** `COOLDOWN_L*`; `stake_regime` EXPLORE/RECOVER; `symbol_loss_cooldown`.

---

## 7. The PlayBook (Bellafiore)

**Insight:** setups nomeados; quando NAO operar; pos-mortem escrito.

**Anti-padrao do LLM:** feature anonima (“melhorar gate”); misturar tres motivacoes num unico PR.

**Regra no Aether:** antes de mudar gate/risco, nomear o setup e o veto; documentar hipotese e criterio de falha.

**Ancoras:** esta doutrina; skill `aether-session-review`; PRs com escopo unico.

---

## 8. Analise Tecnica dos Mercados Financeiros (Murphy)

**Insight:** indicadores sao contexto/filtro, nao oraculo.

**Anti-padrao do LLM:** subordinar TCN cegamente a RSI; ou ignorar path micro adverso.

**Regra no Aether:** hierarquia TA×TCN explicita (price_zone, discordance, adverse_micro_path).

**Ancoras:** `execution_price_zone_gate.py`; `execution_direction_discordance.py`; `execution_adverse_path_gate.py`; RSI/ADX em indicator gating.

---

## 9. Quando os Genios Falham (LTCM / Lowenstein)

**Insight:** modelo ≠ mercado; correlacao sob stress; leverage mata.

**Anti-padrao do LLM:** remover caps “temporariamente”; alongar Redis timeout sem motivo; desligar settlement queue.

**Regra no Aether:** preservar fail-safes de stake, settlement e estado.

**Ancoras:** `max_safe_stake_cap` / `max_safe_stake_pct`; `settlement_queue_ops.py`; Redis timeouts; locks de orquestracao.

---

## Diagnostico de sessao live

Ler o log nesta ordem (processo primeiro, P&L depois):

1. **CLUSTER** — Prob / Cal / Margin / Edge. Margin fraca vs `signal_skip.min_direction_margin` (**0.022**) → soft Kelly (`cal_margin_soft`); nao ha `hard_cal_margin_floor` (removido). Com `neg_edge_hard_skip` **true** (SSOT), Edge `<= 0` ou Edge `< min_edge_*` (**0.015**, inclusive subfloor positivo) deve ser EXEC_EMPTY `neg_edge` / `neg_edge_subfloor_hard` (`gate_verdict=HARD_SKIP`); Soft_SIZE so de soft flags **com** Edge >= floor. Regressao se Edge Cal TCN `< floor` ainda EXEC, Soft_SIZE sob Edge subfloor, ou se Single-Strike/CAP sobe stake sob `SOFT_SIZE`.
2. **SIDE_EQ** — bias e N; soft Kelly sizing apenas (nao SKIP de direcao; nao e licenca para forcar trade).
3. **IND** — RSI/ADX/HURST/ATR/BBW como telemetria de contexto (vetos de sinal removidos).
4. **KELLY** — `mode=explore|recover`, `live_n`, `f*`, `kelly_fraction_scale` (inclui SIDE_EQ soft). Com fusao, `p` rastreia `fusion_p_eff` **so se** Cal TCN ja passou `neg_edge` (nao breakeven se `p_eff` alto). Cold start: `explore_stake_scale_floor` **0.40**. Piso stake **1%** banca; Soft_SIZE **2.5%** so se Edge>=0.015 (e Soft_SIZE so existe com Edge >= floor). `cover_enabled` **false** — PEND nao infla via cover amort. Se EXEC com stake &lt; 1% → regressao.
5. **EXEC / EXEC_EMPTY / EXEC_PAUSE** — `gate_reason` tecnico coerente = processo ok.
6. **RESOLVED / RISK** — WIN/LOSS atualiza pending; PUT loss ≠ rearmar quality gate; nao reescrever knobs por um ciclo.

Narrativas proibidas no diagnostico: “estava quente”, “o mercado deve reverter”, “precisamos operar mais para aprender”.

---

## Checklist rapido antes de mudar codigo

| Pergunta | Livro |
|----------|-------|
| A mudanca aumenta trades sem subir evidencia? | Taleb / Mlodinow |
| Ha hipotese falsificavel e metrica? | Ellenberg / PlayBook |
| Julgamos processo ou so P&L? | Duke / Douglas |
| Caps e settlement permanecem? | Bernstein / LTCM |
| TA vira oraculo ou filtro? | Murphy |
