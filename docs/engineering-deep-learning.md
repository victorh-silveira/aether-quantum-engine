# Deep Learning e meta

Guia operacional DL para agentes. Detalhe de features: [`arquitetura.md`](arquitetura.md) §4–5. Host CUDA / sidecars / event loop: [`engineering-architecture-senior.md`](engineering-architecture-senior.md).

## Runtime atual (SSOT settings)

O `launch-train` separa treino de deploy. Checkpoint TCN exportado com
`deploy_ok=false` alimenta o treino meta e pode ser carregado localmente para
operacao limitada em qualquer tipo de conta, mas nao e enviado ao MinIO. O
gate de qualificacao continua avaliando ACC, Brier e Wilson OOS, mas o
proxy M5 nao concede qualificacao sem evidencia broker/tick.
O meta sem qualificacao OOS fica em `data/dl/meta_candidate.joblib`; o sidecar
usa somente `infra/docker/meta-models/meta_lgbm.pkl`. Falhas de dados, geometria,
treino ou exportacao continuam encerrando o script com erro. A mudanca para
`spot_forward` invalida checkpoints anteriores quando o label diverge.
Um checkpoint local tecnicamente valido que reprovou o gate pode gerar ordens
em DEMO e REAL pelo mesmo caminho de decisao e abertura. O teto soberano por
ordem e 0,1% da banca; `model_deploy_qualified=false` permanece na telemetria.
Checkpoint ausente ou incompativel continua bloqueado. Esse modo nao comprova
vantagem preditiva nem deve ser confundido com promocao estatistica.
As ultimas `deploy_gate.mini_bars` velas ficam fora do ajuste TCN, da selecao
de epoca e da calibracao; sao usadas apenas na simulacao de settlement. O
teste de settlement reconstrói indicadores com as mesmas
`inference_history_bars` velas do runtime (384 no SSOT), sem olhar o futuro. O
carregamento Timescale para meta filtra epochs nao alinhados a M5 antes de
selecionar as 5000 velas, evitando misturar candles parciais do motor.

O experimento TCN de 1HZ75V usa 25 mil M5 e lookback 64; a feature Hurst
passou a estimar o expoente pela inclinacao log-MSD nas escalas 1/2/4/8, em
janela causal de 100 barras. Ela e uma **feature**, nao uma licenca para
operar: no historico anterior, o MSD global deu H≈0,495, e H>0,55 teve
apenas 124 casos recentes com 55,6% de acerto direcional simples. Nao ha
veto arbitrario de entropia/Hurst nem promocao automatica com base nessa
estatistica. O label continua sendo o close no vencimento M5; triple barrier
de primeira colisao nao representa o payoff Rise/Fall deste contrato.

No ensaio local de 23/09/2026 com 25.000 velas M5 e lookback 64, o TCN
encerrou com `val_acc=0.5156`, `val_brier=0.2500`, `settle_wr=19/48=0.3958`,
`settle_lcb90=0.2883` e `settle_brier=0.253`: **nao qualificou para deploy**.
O meta treinou (`OOS IR=0.45`, `z=0.013`), sem alterar o resultado OOS do TCN.
Esse ensaio unico nao isola causalmente o efeito de historico, lookback e
Hurst; nao se deve inferir vantagem nem relaxar o gate a partir dele.

### Auditoria de entrada e settlement (1HZ75V)

`infra.timescale.capture_enabled=true` liga apenas o writer Timescale, mesmo
com `infra.enabled=false`; nao liga Redis/MinIO. O stream existente assina
ticks de 1HZ75V e os grava em `ticks` enquanto o motor estiver ligado. Nao
existe backfill automatico de 1 milhao de ticks; a retencao atual e 30 dias.
Para coletar sem abrir trades, use
`python app/scripts/operations/collect_public_ticks.py` (WSS publico, sem
token, reconexao automatica; Ctrl+C encerra). O smoke
`--max-ticks 3 --max-retries 3` gravou 3 ticks reais em 23/09/2026; isso nao
constitui base historica suficiente para o gate.
`contract_executions` registra compra e settlement, separando `broker`,
`profit_table` e `inferred_rest`. Spots ausentes ficam `NULL`, nunca sao
preenchidos pelo close M5 ou pelo spot de proposta. A view
`contract_label_audit` cruza somente contratos `broker` com spots, tempos,
lucro e velas M5 disponiveis. Exemplo de consulta somente leitura:

```sql
SELECT count(*) AS n,
       avg((m5_win <> broker_win)::int) AS disagreement,
       avg(broker_win::int) AS broker_wr,
       avg((spot_win <> broker_win)::int) AS spot_vs_profit_disagreement,
       avg(broker_brier) AS broker_brier
FROM contract_label_audit WHERE symbol='1HZ75V';
```

A view esta vazia ate haver contratos auditados. O backtest TCN por close M5
agora e explicitamente `m5_close_proxy`: continua diagnostico, mas
`deploy_gate.require_broker_settlement=true` impede que ele ou um checkpoint
antigo sem fonte auditada sejam promovidos. A view nao promove modelo nem
estima latencia artificialmente; falta integrar amostras auditadas suficientes
ao avaliador OOS antes de voltar a qualificar qualquer TCN.
Antes de substituir o gate e necessario coletar ticks completos, verificar
o alinhamento de timestamps e payout efetivo, e repetir OOS purgado. `DEMO`
e `REAL` passam pelo mesmo codigo de captura; a fonte `inferred_rest` nunca
conta como observacao de spot confirmado.

Em uma serie binaria, a Lei dos Grandes Numeros converge para a probabilidade
verdadeira de CALL, nao necessariamente para 50%. Mesmo com labels 50/50,
predizer sempre um lado nao gera vantagem. Para payout liquido de 0.85, o
breakeven de Rise/Fall e `1 / (1 + 0.85) = 0.54054`; 50% implica retorno
esperado de `0.5 * 0.85 - 0.5 = -0.075` por unidade apostada. A janela de
previsao do modelo e uma vela M5, igual ao vencimento do contrato; trocar a
janela altera o evento alvo e requer validacao e contrato operacional novos.

| Item | Valor tipico |
|------|----------------|
| Simbolo | **1HZ75V** (Volatility 75 (1s)) |
| Arch | TCN |
| Lookback | **64** → tensor `[1, 64, 14]` |
| MACRO OHLC | **86400 s** (D1, 365 velas de treino, `data_handler.granularity`) |
| MICRO OHLC | **300 s** (M5, `training_history_bars` / `micro_history_bars` **25000** no treino; inferencia live >= `causal_norm_window` **288** + lookback **64** + 16) |
| Contrato | **5 m** RISE_FALL (ops fixo M5); label TCN **N=1** vela M5 (`spot_forward`) |
| MINI OHLC | **300 s** (`mini_granularity`) |
| Bootstrap wait | `bootstrap_history_wait_cap_seconds` **30** (nao dorme a granularidade inteira entre retries) |
| MILI | Tick flow (nao OHLC) |
| Features | **14D** (`FEATURE_DIM`) |
| Label | `spot_forward` (SSOT settings; CALL se close M5 seguinte > close atual; proxy do settlement, nao spot executado) |
| Fetch treino micro | `max(fetch_count, micro_fetch_count, training_history_bars)` → **25000** velas reais e alinhadas (nao smoke/flat); disponibilidade confirmada no WSS publico |
| Online training | **false** (DEMO usa checkpoint do `launch-train`) |
| ACC / deploy | `force_ok=true` exporta apenas para diagnostico; mini walk-forward M5 e proxy e nao promove com `require_broker_settlement=true`; futura evidencia auditada exige ACC anti-colapso >=**0.50**, Brier **<0.260** e Wilson 90% >= breakeven (**0.540541** para payout 0.85) |
| Limiares live | TCN sempre CALL se Cal ≥**0.5** senao PUT; `apply_calibrator_stable` prefere raw se mais nitido; clamp `[raw±0.05]`; `temperature_min` **0.75**; `min_oos_sharpness` / `min_calibration_sharpness` **0.0** (sem piso de export); banda `[0.45, 0.55]` so telemetria/`raw_extreme` |
| Rodadas | uma tentativa por conjunto de dados; reprova e aguarda dados novos, sem retreinar aleatoriamente sobre o mesmo historico |
| Early stop | `min_epochs` **15**, `early_stopping_patience` **12** |
| Meta | LightGBM **23D** `predicted_payoff_edge` |

Leitura operacional (sessao live): checkpoint `data/dl/1HZ75V.pth` com `val_accuracy` colado ao piso implica Cal mole. Nao “operar mais para aprender”; se ACC estruturalmente no piso, retreinar via `launch-train`. Loss-clf FLIP so apos bootstrap live `LOSS_BOOTSTRAP_EXIT_N` **4**; Edge EV usa payout configurado ate a primeira proposta e taxa liquida cotada na sessao depois disso; Edge ≤ 0 em EXPLORE gera `SKIP:neg_edge`; meta nao flipa lado.

### Frente: vies TCN monotono + ACC no piso

Abrir retreino TCN (skill `aether-dl-train` / `launch-train`) quando **todas** as condicoes abaixo persistirem por sessao:

1. `ACC` / `val_accuracy` colado ao piso (~**0.53**) no `[IND]`
2. TCN ancora o mesmo lado na maioria dos ciclos (ex.: CALL continuo com Cal≥0.5)
3. Alta taxa de `adapted=1` com `why=tape_vs_tcn` ou `candle_vs_tcn` (SCALE corrigindo a ancora)
4. Sem confundir com FLIP ilegítimo: apos `candle_holds`, FLIP+vela==TCN deve aparecer como `blocked=candle_holds`, nao como EXEC invertido

Acao: `launch-train` → gate deploy/settle → `make docker-rebuild` + sync MinIO. **Proibido** abrir gate IND (RSI/ADX) ou Soft Kelly do loss-clf para “corrigir” o vies.

## Entry points

| Comando | Papel |
|---------|-------|
| `train.py` / `app/train.py` | treino TCN |
| `app/scripts/batch/launch-train.bat` | sanitize → sweep horizonte N (H1–H4 em M5) + promote → gate → Timescale → meta |
| `app/scripts/operations/run_launch_train_tf_pipeline.py` | orquestra sweep horizonte N + promote (fallback `train.py` se `horizon_sweep.run_in_launch_train=false`) |
| `make docker-rebuild` | recarrega meta/loss apos o treino (**nao** apaga `data/dl`) |
| `app/scripts/operations/sanitize_fresh_run.py` | limpa `data/dl`, meta/loss pkls e estado em `data/` (so train/reset) |
| `app/scripts/operations/check_dl_deploy_gate.py` | `force_ok=true` preserva o artefato para diagnostico, mas nao promove `deploy_ok`; promocao exige geometria e qualidade fora da amostra; simbolos de `settings.symbols` |
| `app/scripts/operations/train_meta_*.py` | treino offline do meta (`--source auto`; `--bars` **5000** em micro M5; nao usar 365 D1) |
| `app/scripts/operations/sweep_train_timeframes.py` | loop de celulas H1–H4; artefactos em `data/dl/sweep/1HZ75V/H{N}`; leaderboard JSON |
| `app/scripts/operations/promote_tf_winner.py` | promove vencedor elegivel para `settings.json` + `drift_symbols.py` + `data/dl` (fail-closed se nenhum) |

## Sweep de horizonte N (launch-train)

O TCN estima deslocamento em **N velas M5**. O `horizon_sweep` e **diagnostico offline** (`enabled` / `run_in_launch_train` **false** no SSOT). Grade **H1–H4** (`n_bars` = 1/2/3/4 em M5) nao promove duracao ops ≠ **5 m** sem mandato. Contrato live permanece `duration=5`.

Pipeline **offline** (nao troca N por ciclo ao vivo):

1. `horizon_sweep.n_bars` / `duration_minutes` no SSOT — celulas **H1…H4** no relogio M5 (lookback/history copiados, sem reescalar wall-clock). Simbolo **1HZ75V**.
2. `run_launch_train_tf_pipeline.py` limpa `data/dl/sweep`, treina cada celula com ckpt isolado (`data/dl/sweep/1HZ75V/H{N}/`), **infra/MinIO off** no sweep, **1 tentativa** (`train_deploy_retries=1`), overlay `logging.level=CRITICAL` se `quiet_train_logs` (falha resumida em `why=` na linha cell), grava leaderboard.
3. Elegivel: **`settle_wr` ≥ be + 0.03** **e** `settle_n ≥ min_settle_n` (**16**) **e** `history_bars ≥ min_history_bars` (**800**). Label ACC e so telemetria.
4. Com `auto_promote=true` (default), promove vencedor: copia ckpt para `data/dl/` (carimba `deploy_ok`) + grava `label_horizon_bars` do winner; **`params.duration`** vem de `ops_contract_duration_minutes` (**5**), **nao** do N do winner — **fail-closed** se nenhum elegivel (meta nao roda).
5. Gate ACC/settle + meta no SSOT promovido. Depois: `make docker-rebuild` + sync MinIO.

Knobs: `horizon_sweep.n_bars` / `duration_minutes` / `ops_contract_duration_minutes` / `quiet_train_logs` / `run_in_launch_train` / pisos settle. Flags CLI: `--only H1 H2`, `--dry-run`, `--skip-promote`.

## Sample weighting (vies de classe + dinamica)

SSOT: `deep_learning.sample_weighting` — modulo `dl_sample_weighting.py`; wired em `train_model_walkforward` via `compose_train_weights` (alinha pesos do purged-split).

| Knob | Padrao | Papel |
|------|--------|-------|
| `class_balance_enabled` | true | Repondera CALL/PUT quando a taxa de label sai do equilibrio |
| `class_balance_eps` | 0.05 | Tolerancia antes de balancear |
| `recency_enabled` | true | Decai peso de amostras antigas |
| `recency_half_life_n` | **500** | Half-life em numero de amostras |

Telemetria de treino: `TrainResult.label_call_frac`, `pred_call_frac`, `minority_recall` (logados em treino bem-sucedido).

## Deploy gate (senior)

`force_ok=true` conserva a exportacao para diagnostico, mas nunca e sinal de qualidade. A acuracia global e somente piso anti-colapso (**0.50**): em uma serie binaria balanceada ela nao prova vantagem. O mini walk-forward de 48 pontos continua calculando Brier e Wilson para diagnostico, mas usa closes M5 e carimba `deploy_settlement_source=m5_close_proxy`. Com `require_broker_settlement=true`, `deploy_ok` e `deploy_provisional_ok` ficam falsos: nao ha promocao baseada nesses numeros. O checker e o runtime tambem rejeitam checkpoints antigos sem fonte `broker_tick_audit`. Ate integrar avaliacao OOS realmente auditada, **nenhum TCN pode qualificar**. O runtime exige checkpoint local tecnicamente compativel em DEMO e REAL; modelo nao qualificado carrega `model_deploy_qualified=false` e teto de stake de **0,1%** da banca. Isso permite observacao operacional, mas nao demonstra vantagem nem elimina perdas. Treino sobrescreve o `.pth` anterior; um checkpoint fraco nao e promovido para MinIO/sidecar, mas pode fornecer probabilidades ao treino meta.

### Modo provisório de observação com capital

`deploy_provisional=true` e distinto de `deploy_ok=true`, mas o modo esta
inacessivel enquanto `require_broker_settlement=true` e o avaliador dispuser
apenas de closes M5. O criterio legado de **120** observacoes, Brier < **0.260**
e taxa bruta >=**0.57** nao autoriza promocao com labels proxy.

Majority-collapse: o gate rejeita pred skew (`|pred-0.5|` / `|pred-label|` > **0.15**) ou label skew com `minority_recall < 0.40`. A telemetria permanece no checkpoint para auditoria.

Checkpoint de treino restaura o pico de **maior val_acc** (sem veto por BCE nem por collapse no pico ACC). Ramo sharp maximiza nitidez entre epocas com ACC≥soft_min e sem `collapse_hit`. Anti-overfit SSOT: `focal_gamma` **1.0**, `weight_decay` **0.01**, `tcn.dropout` **0.25**, `label_smoothing` **0.0**, `lr_scheduler` **reduce_on_plateau**, `aux_regression_weight` **0.08**. Export sharpness: `min_oos_sharpness` **0.0** (assert no-op); temperature sharpen ainda roda se calibracao pedir.

## Meta — alvo e dados


- Alvo preferencial: payoff assinado **cru** ate o split cronologico; winsorize 1%/99% e unit-scale O(1) usam **so o fold de treino** (`label_scale` = std do treino apos clip; `train_std` / `val_std` no bundle). Pontos brutos 1HZ75V nao entram no booster. Prefixo `|payoff|~0` e cortado quando possivel; split degenerado, z-IR sem edge ou `val_mae/train_mae > 1.50` bloqueiam exportacao do meta. Os pisos sao z-score **0.01** e IR **0.05**. Objetivo **regression_l1**; n grande: **80** rounds, early-stop **L1** patience **15**, `max_depth` **1**, `num_leaves` ate **4**, `bagging_fraction` ligado.
- Hydrate Docker = smoke (500/365). `launch-train` chama `ensure_timescale` (seed Deriv) antes do meta: piso micro **5000** / macro D1 **365**. Timescale smoke/curto/flat → INFO e Deriv (nao WARNING "rejeitado"); apos Deriv, seed no Timescale.
- Qualidade OHLC antes do meta: timestamps estritamente sequenciais na granularidade M5, arrays alinhados, precos finitos e positivos, invariantes `low <= open/close <= high` e diversidade minima. Falha no Timescale busca Deriv; falha no Deriv aborta o treino.
- Fit do calibrador: se std calibrado colapsa vs raw no holdout/val → persiste `identity`. Teacher meta: raw+expand em INFO se cal ainda esmagar.
- `validate_target_variance` inclui `source`, `forward_var`, `close_nunique`, `label_scale`. Escala O(1) nao usa o y completo (lookahead no alvo e proibido).

## Calibracao: `raw_extreme` (anti-override)

Modo legado `tcn_macro_override` foi substituido por `raw_extreme` em `dl_calibration_tolerance.py`:

- Se `raw` > `tcn_macro_call_override` ou `raw` < `tcn_macro_put_override`, o modo vira `raw_extreme`.
- No ramo `raw_extreme`, **Cal nao e substituido por raw**; retorno mantem probabilidade calibrada (Kelly usa essa Cal).
- `apply_calibrator_stable` (antes do ramo) devolve **raw** se `|raw-0.5| > |Cal-0.5|`.
- Com Cal na banda neutra (`calibration_neutral_drift` **[0.45, 0.55]**; drift degenerado `[0.5,0.5]` rejeitado), o **lado** segue o limiar raw (`raw_dir`), nao Cal≥0.5.
- Kelly / sizing usam **Cal** (pos-stable), nao o raw cru do TCN.
- Nomes das chaves SSOT (`tcn_macro_*_override`) sao historicos: limiam extremo de **raw TCN**, nao o timeframe MACRO OHLC.

`calibration.method=auto` + `min_calibration_sharpness` / `min_oos_sharpness` (**0.0**): export nao bloqueia por nitidez; se temperatura/Platt/isotonico colapsar nitidez, o fit ainda pode cair para `identity` (raw). `min_calibration_margin_floor` **0.05** no live: se Cal mole e raw nitido, devolve raw; se ambos moles, fica o mais nitido **sem stretch** (nao inventa Edge). Edge CLUSTER = EV; Edge≤0 em EXPLORE → SKIP `neg_edge` (waive com PEND material). Com `online_training` **false**, limpar `EXEC_EMPTY` exige **retreino + export** (sempre no launch-train com `force_ok`). Knobs live sozinhos nao limpam a sessao.

### Ops apos launch-train (obrigatorio)

1. `launch-train` / treino TCN `1HZ75V` com `force_ok` (exporta sempre)
2. Deploy do novo checkpoint / TorchScript no MinIO (`data/dl` + sidecars)
3. Reiniciar o motor

Ate la, `EXEC_EMPTY` + `SKIP:neg_edge` com Cal~0.52 continua correto (`skip_neg_edge` intacto). **Proibido** Soft Kelly no TCN ou desligar `skip_neg_edge` para “passar” trade.

Live: `clamp_calibrated_call_to_raw_band` clipa **p_call** em `[raw±max_calibrated_raw_gap]` (**0.05**) **antes** de `apply_calibration_neutral_tolerance`, para CLUSTER e Kelly usarem o mesmo Cal. PUT espelha `1-p_call`. `min_calibration_margin_floor` **0.05** (= half-width) e fallback; o restore principal e margem relativa raw vs Cal. Lado live: CALL se Cal ≥**0.5**, PUT se Cal <**0.5** (sem `SKIP:NEUTRAL_ZONE`). Limiares `confidence_call_threshold` **0.57** / `confidence_put_threshold` **0.43** nao skipam. Banda `calibration_neutral_drift` **[0.45, 0.55]** so telemetria e ramo `raw_extreme` (lado raw quando Cal mole). Metricas: `cal_raw_gap_capped` / `cal_raw_gap`. Isso alimenta Edge/Kelly (nao so `trade_score`). `temperature_min` **0.75** permite T&lt;1 no fit.

Fusao: `why=tcn_pos_edge` exige Cal **e** raw_edge ≥ `fusion_min_edge_execute` (**0.04**). Sintoma de regressao: CLUSTER Prob≈BE + `p_put`≫0.70 + `why=tcn_pos_edge` com `raw_edge`~0.

Visao multi-escala (MACRO/MICRO/MINI/MILI) e soft Kelly ficam fora do pacote DL — ver [`engineering-orchestrator.md`](engineering-orchestrator.md) e `orchestrator.execution.scale_vision`.

## Pos-migrate hibrido (legado → SSOT atual M5)

1. Invalidar checkpoints `data/dl/*.pth` e TorchScript MinIO com `granularity`/`lookback` ≠ settings (ex.: legado **180**/7200 M3, **120**/3600 ou lookback **720** / M1 **60**/7200).
2. Verificar Timescale com `docker-hydrate.sh` e popular OHLC **real** com `ensure_timescale.py` para micro/MINI **300** / macro **86400**.
3. Retreinar com **`launch-train.bat`** (TCN `lookback=64`, micro **300**, label `spot_forward` N=1; contrato ops **5 m**) + meta — **nao** via `launch-all-demo`.
4. So depois: `launch-all-demo.bat`; validar CFG live `ohlc=300s`, `macro=86400s`, `contrato=5 m`, `label_horizon_bars=1`.

Com `online_training=false` (SSOT), a DEMO nao agenda retreino TCN em runtime (nem settle nem rolling); usa o checkpoint do `launch-train`. Para reativar, `online_training=true` + `rolling_retrain_bars` / `retrain_min_bars` (sem `mark_force_retrain` no settle). Meta e loss-clf fazem `/v1/learn` a cada trade.

## Pacote

`app/src/application/services/deep_learning/` — features, labels, predict, calibracao, deploy, checkpoint.

## Meta runtime

- Inferencia TCN: eager/CUDA local no host (`data/dl`)
- Meta HTTP: `aether-meta-classifier`; artefato em `infra/docker/meta-models/`

## Anti-padroes

- Trocar `label_mode` sem retreinar
- Desligar `force_ok` sem mandato (bloqueia launch-train)
- Treinar meta em OHLC sintetico/flat do hydrate Docker
- Soft Kelly no TCN / desligar `skip_neg_edge` para “passar” Edge mole

Skill: `aether-dl-train`.
# Contrato temporal dos indicadores

As series de ATR e largura de Bollinger usam z-score rolling causal nas janelas
configuradas de cada indicador. Acrescentar candles futuros nao pode modificar
os valores ja calculados. RSI, ADX e Keltner mantem valores neutros durante seu
warmup, sem preencher o passado com estatisticas de barras posteriores.

`atr_raw` e ATR relativo ao fechamento; `atr_abs` e ATR em unidades de preco;
`atr_norm` e z-score. Gates que comparam amplitude de candle usam `atr_abs`.
O retorno auxiliar de treino parte do fechamento da mesma barra que encerra
a sequencia e usa o horizonte e a suavizacao forward configurados no label.
Nao ha interpolacao de precos futuros nessa montagem.

Alteracoes destas transformacoes exigem retreino e revalidacao dos artefatos
TCN/meta/loss afetados; manter 14 colunas nao garante compatibilidade semantica.
