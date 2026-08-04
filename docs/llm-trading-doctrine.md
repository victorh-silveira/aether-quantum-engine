# Doutrina LLM do Aether (9 livros)

O LLM/Cursor e **copiloto de engenharia e auditoria**. Nao decide CALL/PUT em runtime. A decisao live permanece TCN + meta LightGBM + Kelly/caps em codigo (vetos de sinal/qualidade removidos por mandato operacional escopo 1).

SSOT operacional: [`config/settings.json`](../config/settings.json). Metodologia: [`medallion.md`](medallion.md). Sample size: [`sample-size-lln.md`](sample-size-lln.md).

---

## 1. Iludido pelo Acaso (Taleb)

**Insight:** streak e path-dependence nao sao edge; o acaso recompensa narrativas.

**Anti-padrao do LLM:** afrouxar piso apos 2–3 WINs; tratar Cal 0.51 como sinal; aumentar stake porque “estava batendo”.

**Regra no Aether:** caps limitam cauda; Cal/Edge sao telemetria de processo (nao veto de codigo apos escopo 1).

**Ancoras:** soft recovery caps; Kelly; `force_trade_every_cycle=false`.

---

## 2. A Caminhada do Bebado (Mlodinow)

**Insight:** N pequeno e ruido; a media so converge com volume.

**Anti-padrao do LLM:** hard-skip ou flip apos 2 losses; misturar `live_wr` com N=1; “aprendemos rapido” via mais trades frios.

**Regra no Aether:** `sample_size_policy` (telemetria/sizing); explore stake scale com N baixo.

**Ancoras:** `sample_size_policy` (`evidence_n_min=20`, `large_n_min=40`, `explore_stake_scale_floor=0.25`); `app/src/domain/analytics/sample_size_policy.py`; [`sample-size-lln.md`](sample-size-lln.md).

---

## 3. Como Nao Errar (Ellenberg)

**Insight:** esperanca, Bayes e taxa-base; mudanca de knob precisa de hipotese falsificavel.

**Anti-padrao do LLM:** alterar threshold “porque o log ficou feio”; ignorar ACC de validacao; misturar prior e evidencia sem shrink; tratar streak de PUT LOSS como motivo para **rearmar quality gate** ou veto de sinal.

**Regra no Aether:** Kelly bayesiano; gate de `val_accuracy`; calib drift so com N minimo. Vies de lado corrige-se em **treino** (`sample_weighting` / majority-collapse) + **sizing** SIDE_EQ soft Kelly — nao reintroduzindo veto de qualidade.

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

**Regra no Aether:** soft recovery amortiza `pending_loss`; stop-win ativo; stop-loss interno desativado por politica — nao reativar sem mandato explicito.

**Ancoras:** `soft_recovery_policy`; `pending_loss`; stop-win composto; `app/src/domain/risk/risk_recovery_state.py`.

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

1. **CLUSTER** — Prob / Cal / Margin / Edge. Cal ~0.50 com Margin &lt; `hard_cal_margin_floor` nao e setup.
2. **SIDE_EQ** — bias e N; soft Kelly sizing apenas (nao SKIP de direcao; nao e licenca para forcar trade).
3. **IND** — RSI/ADX/HURST/ATR/BBW como telemetria de contexto (vetos de sinal removidos).
4. **KELLY** — `mode=explore|recover`, `live_n`, `f*`, `kelly_fraction_scale` (inclui SIDE_EQ soft). Cold start deve ter stake comprimida.
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
