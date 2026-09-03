# Doutrina LLM do Aether (9 livros)

O LLM/Cursor e **copiloto de engenharia e auditoria**. Nao decide CALL/PUT em runtime. A decisao live permanece TCN + meta LightGBM + Kelly/caps + **fusao EV multi-escala** (`execution_direction_fusion`); **escopo 1.1** + arquitetura continua 1HZ75V: catálogo `signal_skip` (`mini_pair_oppose` / `cal_margin` / chop = soft Kelly **0.55**); **neg_edge** le Edge Cal TCN do lado (`CLUSTER`, nao `fusion_p_eff`); soft se `0 < Cal < min_edge_execute` (`neg_edge_hard_skip` **false**, `neg_edge_soft_min_edge` **−1.0**); **hard** se Cal `<= 0` (`fusion_p_eff` nao libera EXEC); **Anti-loss** com microestrutura M5 (ancora hibrida ops N=3 + ultima vela fechada; EMA slope 2-pontos 9/21; RSI 0.35/0.65; cache EMA por ciclo). `invert_exec_side` **false**; `online_training` **false**; stop-win **4.31%** Kelly Single-Strike.

SSOT operacional: [`config/settings.json`](../config/settings.json) — universo **`1HZ75V`** (Volatility 75 (1s) Index), contrato ops **5 m (M5)**, label TCN **N=1** vela M5 (`quantum_multi_barrier`), micro/MINI **300 s** (M5), ciclo **120 s**, macro **86400 s** (D1). Metodologia: [`medallion.md`](medallion.md). Sample size: [`sample-size-lln.md`](sample-size-lln.md). Loss ML: [`infra-docker.md`](infra-docker.md) (`aether-loss-classifier`). Arquitetura senior (host 3.13 / DDD / sidecars): [`engineering-architecture-senior.md`](engineering-architecture-senior.md).

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

**Regra no Aether:** Kelly bayesiano; gate de `val_accuracy`; calib drift so com N minimo. Vies de lado: treino + SIDE_EQ soft Kelly. Catalogo sinal (`signal_skip`) = soft Kelly para mini/cal/chop; loss-clf = soft em `[veto_p_loss_floor, hard_p_loss_floor)` com `veto_ready` e **FLIP** CALL↔PUT **relativo ao TCN** se `p_loss >= hard_p_loss_floor` (**0.90**, `veto_ready`); seed bloqueia FLIP contra vela; chop = soft Kelly continuo; **neg_edge** soft se `0 < Cal TCN < min_edge_execute` (`neg_edge_hard_skip` **false**); hard `gate_reason=neg_edge` se Cal TCN `<= 0` (`fusion_p_eff` nao lava), seed+edge profundo ou override. Kelly usa `fusion_p_eff` so apos esse gate.

**Ancoras:** `risk_management.min_validation_accuracy_gate` (0.53); `soft_min_val_accuracy`; `deep_learning.sample_weighting`; `deploy_gate.reject_majority_collapse`; `execution_side_eq_sizing`; `apply_live_calib_drift_soft`; Kelly em `app/src/domain/risk/`.

---

## 4. Pensando em Apostas (Duke)

**Insight:** qualidade do processo > resultado do ciclo.

**Anti-padrao do LLM:** `force_trade_every_cycle=true` como “fix” de EXEC_EMPTY; julgar sessao so pelo P&L de 5 trades.

**Regra no Aether:** `EXEC_EMPTY` com `gate_reason` coerente e sucesso de processo; force trade permanece off.

**Ancoras:** `force_trade_every_cycle: false`; logs `gate_reason` / `quality_gate_reason`; `execution_quality_reject.py`.

---

## 5. Desafio aos Deuses (Bernstein)

**Insight:** risco e o que se mede e limita; incerteza sem medida nao escala exposicao.

**Anti-padrao do LLM:** reinventar stop-loss interno; recovery sem pending; sizing sem teto de banca.

**Regra no Aether:** soft recovery usa amortização equilibrada de `pending_loss` (amort **2/3**, `cover_multiple` **1.10**, caps L0/L1 **3.5%**); stop-win ativo; damping de proximidade da meta (`target_damping_*`) comeca em **1.0** e cai ate **0.50**; stop-loss interno desativado por politica — nao reativar sem mandato explicito. EXPLORE forçado (`neg_edge_soft` / near_stop / quality / `f*≈0`) usa piso `neutral_bankroll_pct` — **nunca** cover pleno sob soft sem PEND (revenge); **nunca** `dlambert_unit` sticky da primeira Kelly boa como tamanho de ordem. Cover inviavel com PEND material → stake = **CAP** (parcial DAL, telemetria `RECOVERY_INFEASIBLE`). Cover so no caminho RECOVER/DAL quando cover ≤ CAP. Soft + PEND material + stake ≈ CAP linear3 sem inviabilidade = regressao cover-pleno.

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

1. **CLUSTER** — Prob / Cal / Margin / Edge. Margin fraca vs `signal_skip.min_direction_margin` (**0.022**) → soft Kelly (`cal_margin_soft`); nao ha `hard_cal_margin_floor` (removido). Edge Cal `<= 0` + EXEC (mesmo com `fusion_p_eff` > be) → regressao: fusao lavou o gate.
2. **SIDE_EQ** — bias e N; soft Kelly sizing apenas (nao SKIP de direcao; nao e licenca para forcar trade).
3. **IND** — RSI/ADX/HURST/ATR/BBW como telemetria de contexto (vetos de sinal removidos).
4. **KELLY** — `mode=explore|recover`, `live_n`, `f*`, `kelly_fraction_scale` (inclui SIDE_EQ soft). Com fusao, `p` rastreia `fusion_p_eff` **so se** Cal TCN ja passou `neg_edge` (nao breakeven se `p_eff` alto). Cold start: `explore_stake_scale_floor` **0.40**. RECOVER: stake = cover (`cover_multiple` **1.10**, amort **2/3**); `f*` so gate. Se `[RESOLVED] PEND` nao cai apos WIN → regressao de audit.
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
