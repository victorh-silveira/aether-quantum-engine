# Doutrina de trading para o LLM (Aether)

Politica autorizada atual: quatro vetos extremos e reavaliacao direcional
conforme [catalogo](engineering-indicator-gates.md). O gatilho nao cria vantagem
estatistica nem modifica P(CALL); aceita candidato somente com EV suficiente
na distribuicao existente. A descricao legada de confluencia e contra-tendencia
abaixo fica subordinada a `four_market_vetoes` quando esse knob esta ativo.

O LLM/Cursor e **copiloto de engenharia e auditoria**. Nao decide CALL/PUT em runtime.

Decisao live: TCN (Cal vs 0.5) → **FLIP** loss-clf se auto_learn, `n_train>=1` e `p_eff` no piso → `invert_exec_side` (**false**) → quatro vetos extremos → Kelly / cover_l0 → proposta com payout cotado, EV positivo e margem de 0,01 sobre break-even. Confluencia adicional e regime sao telemetria; ruido/tendencia/conformal/compressao nao somam vetos no SSOT. SKIP tecnico: treino/dados/checkpoint/predict/stop-win. `acc_floor` so se knob `skip_below_soft_min_acc` **true** (ops **false**). Sem `cal_soft_edge` / quality gate amplo / skips de vela/scale/doji.

**Removido:** HARD SKIP por loss-clf, Soft Kelly do loss-clf ou META, fusao EV, signal_skip soft legado, skips de vela/scale/doji no path live, micro/regime/vol/exhaust generico, anti-loss EMA/RSI, `cal_soft_edge`. `skip_neg_edge=false` evita veto duplicado; a proposta final ainda exige EV e margem.

Universo: **1HZ75V** M5 (contrato **5 m**; label N=1; ciclo **300 s**; payout **0.85**; stop-win **4.31%** Single-Strike).

Ops: `invert_exec_side` **false**; `skip_neg_edge` **false**; `skip_exec_vs_candle` / `skip_scale_candle_discord` / `skip_doji` **false**; `adapt_retract_enabled` **false**; `skip_below_soft_min_acc` **false**; cover `amort_cycles` **1/1** + cap L0.

## Nunca propor

1. `force_trade_every_cycle=true` como fix de EXEC_EMPTY
2. Reabrir quality gate amplo / `signal_skip` multi-gate / Soft do loss-clf / HARD SKIP no lugar do FLIP no piso
3. Revenge sizing apos LOSS; subir amort acima de 1 sem mandato
4. Remover caps, settlement ZSET ou timeouts “temporariamente”
5. Julgar mudanca so pelo P&L de poucos ciclos

## Sempre fazer

1. Distinguir EXPLORE vs RECOVER; `cover_enabled` **true** (`cover_multiple` **1.0**, amort **1** → `min(max(PEND/payout, 1% banca), cap_L0)`); cover L0 **3.5%** subordinado ao teto inicial do checkpoint TCN de **1% por ordem**; PEND nao force-explore por near-stop
2. SKIP tecnico ou economico = processo coerente quando dados, checkpoint ou EV da cotacao falham; FLIP young `pe>=0.58` / mature `pe>=0.70` apos auto_learn e `n_train>=1`. Cal~0.52 com payout 0.85 nao implica EV positivo; retreino nao garante edge.
3. Evidencia: `live_n`, Cal/Edge (EV vs BE), `val_accuracy`, telemetria `LOSS_CLF` (`p=` / `pe=` / `floor=` / `boot=N/2` / `n=`) e `QUALITY` pos-settle se houve FLIP
4. Recovery prioriza amortizar PEND dentro do teto de stake; a cotacao final continua exigindo EV positivo e margem.
