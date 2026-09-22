# Doutrina de trading para o LLM (Aether)

O LLM/Cursor e **copiloto de engenharia e auditoria**. Nao decide CALL/PUT em runtime.

Decisao live: TCN (Cal vs 0.5) → **FLIP** loss-clf se auto_learn, `n_train>=4` e `p_eff` no piso (sem trava por vela) → `invert_exec_side` (**false**) → SKIP `neg_edge` em EXPLORE (**waive** somente com PEND ≥ `material_pending_min`) → Kelly / cover_l0 (META inerte / no-op no sizing). Confluencia de indicadores e telemetria: nao sobrescreve o lado. O edge usa o payout configurado ate a primeira proposta valida e a taxa liquida efetivamente cotada na sessao depois disso. SKIP tecnico: treino/dados/deploy/predict/stop-win / cooldown / pausa. `acc_floor` so se knob `skip_below_soft_min_acc` **true** (ops **false**). Sem `cal_soft_edge` / quality gate amplo / skips de vela/scale/doji.

**Removido:** HARD SKIP por loss-clf, Soft Kelly do loss-clf ou META, fusao EV, signal_skip soft legado, skips de vela/scale/doji no path live, micro/regime/vol/exhaust generico, anti-loss EMA/RSI, `cal_soft_edge`. **Edge TCN ≤ 0 = SKIP em EXPLORE** (`skip_neg_edge`); waived com PEND material.

Universo: **1HZ75V** M5 (contrato **5 m**; label N=1; ciclo **300 s**; payout **0.85**; stop-win **4.31%** Single-Strike).

Ops: `invert_exec_side` **false**; `skip_neg_edge` **true**; `skip_exec_vs_candle` / `skip_scale_candle_discord` / `skip_doji` **false**; `adapt_retract_enabled` **false**; `skip_below_soft_min_acc` **false**; cover `amort_cycles` **1/1** + cap L0.

## Nunca propor

1. `force_trade_every_cycle=true` como fix de EXEC_EMPTY
2. Reabrir quality gate amplo / `signal_skip` multi-gate / Soft do loss-clf / HARD SKIP no lugar do FLIP no piso
3. Revenge sizing apos LOSS; subir amort acima de 1 sem mandato
4. Remover caps, settlement ZSET ou timeouts “temporariamente”
5. Julgar mudanca so pelo P&L de poucos ciclos

## Sempre fazer

1. Distinguir EXPLORE vs RECOVER; `cover_enabled` **true** (`cover_multiple` **1.0**, amort **1** → `min(max(PEND/payout, 1% banca), cap_L0)`); cover L0 **3.5%**; PEND nao force-explore por near-stop; piso **1%** soberano
2. SKIP tecnico / `neg_edge` = processo ok quando coerente (EXPLORE); FLIP young/mature `pe>=0.58` (apos auto_learn e `n_train>=4`); Cal~0.52 + Edge≤0 = `neg_edge` correto — limpar exige **retreino + export** (`force_ok`), nao Soft Kelly no TCN
3. Evidencia: `live_n`, Cal/Edge (EV vs BE), `val_accuracy`, telemetria `LOSS_CLF` (`p=` / `pe=` / `floor=` / `boot=N/4` / `n=`) e `QUALITY` pos-settle se houve FLIP
4. Com PEND material, `neg_edge` e waived; recover prioriza zerar divida (skips de vela/scale/doji desativados no hot path)
