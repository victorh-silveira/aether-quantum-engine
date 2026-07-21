# Metodologia quantitativa

O Aether Quantum Engine herda a postura **Medallion** no sentido operacional: o mercado é um **sistema de sinais ruidosos**, não uma narrativa macro discricionária. A implementação concentra-se no índice **`R_10`** com **Deep Learning** e classificação binária Rise/Fall.

Para arquitetura de código, ver [`arquitetura.md`](arquitetura.md).

---

## 1. Princípios

| Princípio | No motor atual |
|-----------|----------------|
| Sinais, não histórias | Direção CALL/PUT estritamente pela TCN (`P(CALL) > P(PUT)`) |
| Horizonte curto | Contexto DL **600 s** (assinatura legado `m15`); execução **120 s** (assinatura legado `m5`); proporção multi-timeframe **1:5** (120:600); label atual `spot_forward` (Triple Barrier / `ma_trend` disponiveis via config) |
| Acoplamento temporal | Inferências e rotações seguem `signature_boundary_seconds` (fallback `cycle_interval_seconds`, padrão **120 s**); fronteira `m5_boundary_epoch` (nome legado) |
| Esteira seletiva | `mandatory_trade_each_cycle: false` + `price_zone` — CALL só em zona compra; PUT só em zona venda; skip no meio |
| Force trade | `force_trade_every_cycle: false` — sem síntese forçada de candidato |
| Modelo pronto antes de operar | `FASE TREINO` suspende ordens até treino da sessão |
| Fail-closed seletivo | Triton obrigatório (`infra.triton.require_for_execution`); meta **opcional** (`require_meta_for_execution: false`) |
| Feedback real | Win rate live misturado em `val_accuracy`; retreino após loss |
| Defesa contra ruído CSPRNG | Consensus Entropy Penalty no Kelly base — **desligado** nos settings atuais (`consensus_penalty_enabled: false`) |
| Persistência financeira | Recovery atrelado a `pending_loss`, não a WIN operacional isolado |
| Soft recovery + caps | `soft_recovery_policy` ativo: amortiza pending em 2–5 ciclos com teto % banca |
| Sizing | EXPLORE = Kelly (`fraction: 0.08`); RECOVER = Soft Recovery amortizado |
| Side equilibrium (LLN) | `side_equilibrium`: small-N hard skip; large-N soft Kelly |
| Meta por sessão ativa | Stop win de 2,60% composto (banca ≥ $100) ou fixo $10 (banca < $100) |
| Sem disjuntor de perda | Stop loss interno desativado |
| Isolamento de estado | `asyncio.Lock` serializa inferência, liquidação e persistência |
| Persistence guard | Após 2 losses na mesma direção: **skip** do candidato (sem flip CALL/PUT); FREEZE em congestão micro |
| Calibração | Zona neutra ON (`neutral_half_width: 0.04`, banda `[0.46, 0.54]`); thresholds **0.54/0.46**; override TCN macro se raw&gt;0.65 ou &lt;0.35 |
| Veto Cruzado TCN-GBDT | Soft veto comprime score; hard veto só com shadow calibrado (`meta_veto_mode`: none/soft/hard); meta opcional para execução |
| Settlement resiliente | Fila Redis `settlement:queue:priority`; tolerância **90 s** |
| Starvation escape | Após **6** quality skips consecutivos, pisos de margem/edge/Z decaem |
| Microestrutura HARD | Vetoes `adx_starvation`, `vol_ratio_starvation`, `val_accuracy_gate` (`min_validation_accuracy_gate: 0.63`) |

---

## 2. Universo Drift e perfil de qualidade

### 2.1 Universo Drift

Índices sintéticos correlacionados no eixo de barreiras. Cada símbolo tem modelo DL independente com **34 features** e volatilidade calibrada ao alvo do índice.

| Símbolo | Papel típico |
|---------|----------------|
| `R_10` | Universo operacional unico; ancora e unico simbolo de treino/execucao |

Operação: contratos **RISE_FALL** de **120 s** (CALL = alta no período, PUT = queda).

### 2.2 Telemetria de Volatilidade, Exaustão e Fluxo Micro

Indicadores micro de **120 s** (RSI, `vol_ratio`, Keltner, `bb_width`, aceleração de ticks, shadow de volatilidade e momentum de spread) alimentam o container `aether-meta-classifier` (porta **8005**) via vetor **43D**, indexados na resolução amostral micro do TimescaleDB. O `LGBMRegressor` (huber) estima `predicted_payoff_edge` contínuo; o resolver preserva score orgânico da TCN quando o edge é positivo e aciona downgrade D-SQUEEZE quando o edge colapsa em microestrutura. Nos settings atuais, meta é **opcional** para execução.

**Spread de convicção cross-symbol** (triplet anexado em `prepare_meta_classifier_cross_symbol_bundle`; zeros no modo single-symbol):

| Feature | Descrição |
|---------|-----------|
| `cross_symbol_prob_delta` | Divergencia de conviccao entre pares (0.0 sem peer) |
| `cross_symbol_vol_ratio_diff` | Spread linear micro de `vol_ratio` (0.0 sem peer) |
| `cross_symbol_rsi_spread` | Spread linear micro de RSI (0.0 sem peer) |

Em regimes de drift paralelo (ambos símbolos com scores altos na mesma direção), spreads baixos sinalizam saturação espelhada — o GBDT usa isso para evitar entradas sem viés relativo.

Features de fluxo e microestrutura extraídas do `TickBuffer` e precomputação:

| Feature | Descrição |
|---------|-----------|
| `micro_tick_acceleration` | Aceleração estocástica de ticks nos últimos 5 s do bloco micro corrente (120 s) |
| `keltner_deviation_ratio` | Distância fracionária do último tick ao centro do canal Keltner micro |
| `micro_bid_ask_spread_momentum` | Taxa de variação de ticks aglutinados por sub-janelas de 5 segundos no bloco micro corrente |
| `micro_bid_ask_spread_momentum_zscore` | Z-Score adaptativo histórico de 1024 períodos da variação de ticks, clipado a ±3.0 |
| `volatility_shadow_ratio` | Razão entre a soma dos pavios (superior + inferior) da barra micro atual e a amplitude do desvio padrão do Keltner (ATR) |
| `volatility_shadow_ratio_zscore` | Z-Score adaptativo histórico de 1024 períodos da razão de pavios, clipado a ±3.0 |

Indicadores macro (Hurst, ADX, bandas) permanecem em `metrics["indicators"]` / `feature_vector` (34D TCN) como telemetria analítica e insumo do stacking — com vetoes HARD de microestrutura em ADX / `vol_ratio` / `val_accuracy` via `execution_quality_gate_microstructure`.

### 2.5 Perfil de qualidade atualizado

| Camada | Comportamento |
|--------|---------------|
| Bloqueio técnico | `data`, `predict_error`, `training`, `deploy_ok=false`, Triton fail-closed |
| Calibração | Zona neutra ON (`neutral_half_width: 0.04`); thresholds **0.54/0.46**; override TCN macro se raw&gt;0.65 ou &lt;0.35 |
| Veto cruzado TCN-GBDT | Z-Score `< -0.20` reclassifica para `NO_EDGE_NEUTRAL`/`LOSS_EXPECTED` mesmo com `WIN_EXPECTED` explícito do meta e força SKIP (`meta_payoff_negative_zscore_veto`); waiver em recovery crítico (`linear >= 4` ou `pending > 150`) com TCN extrema (`raw_prob <= 0.22` PUT / `>= 0.78` CALL) |
| Classificação macro | TCN processa lookback de ~12 h em barras de **600 s** (`[1, 72, 34]`); define direção (`dl_direction`) |
| Stacking tabular | Meta-regressor LightGBM (micro **120 s**) sobre vetor **43D** + probabilidade TCN; saída `predicted_payoff_edge`; meta **opcional** para execução |
| Z-Score de payoff | `payoff_edge_zscore`: janela adaptativa 15–45; classificação estatística do micro-edge |
| Scoring de ranking | `market_decision_score = tcn × max(0.1, 1 + z)` — prioriza Z-Score favorável sem rótulos categóricos |
| Scoring direcional | TCN define `dl_direction`; edge `> 0` mantém score orgânico; veto Z negativo aborta antes do D-SQUEEZE; compressão BB severa rebaixa para `0.52` |
| Margem direcional | `direction_margin = abs(P(lado_escolhido) − 0.50)`; thresholds **0.54/0.46**; `min_direction_margin: 0.03` como **hard gate** (`direction_margin_gate`) |
| Gate de qualidade | Dual soft TCN+meta + **HARD** microestrutura (`adx_starvation`, `vol_ratio_starvation`, `val_accuracy_gate`); sniper/Hurst/BB são stubs (`False`); `require_meta_for_execution: false`; consensus **off** |
| Indicator gating | enabled; `adx_min` 0.20; `vol_ratio_min` 0.65; `veto_on_noise` false |
| Persistence guard | Após 2 perdas na mesma direção: **skip** (`persistence_guard_skip`); flip CALL/PUT **suprimido** em produção; `FREEZE` em congestão |
| Rotulagem | Padrão `spot_forward`; `ma_trend` / Triple Barrier disponíveis via config |
| Perda TCN assimétrica | Penalidade 2,5× para erro direcional em alta volatilidade |
| Optuna meta | Maximiza Information Ratio; constraint OOS payoff Z-Score ≥ +1,00 |
| Gerenciamento de risco | Kelly EXPLORE + Soft Recovery RECOVER (`soft_recovery.enabled: true`, `kelly.fraction: 0.08`, tetos 3,5%); `loss_protection.min_direction_margin: 0.03` |
| BB squeeze adaptativo | `bb_width_adaptive_squeeze.enabled: false` |
| Dynamic threshold | `dynamic_threshold.enabled: false` |

---

## 3. Blindagem multi-timeframe

**Invariante 1:5:** o relógio operacional micro (`data_handler.micro_granularity` = **120 s**) e o contexto macro DL (`data_handler.granularity` = **600 s**) mantêm proporção fixa **1:5**. Cada bloco macro cobre exatamente cinco fronteiras micro; a assinatura `m5b:{boundary};m5:{sym}@{epoch};m15:...` (prefixos **legados**) e `seconds_until_next_signature_boundary` ancoram espera e invalidação de cache na virada cheia de **120 s**.

| Camada | Timeframe | Papel |
|--------|-----------|-------|
| Deep Learning / TCN | Macro **600 s** | Tensor `[1, 72, 34]` ≈ 12 h de contexto macro |
| Meta-regressor GBDT | Micro **120 s** | Regressão tabular **43D**; edge contínuo + downgrade D-SQUEEZE |
| Orquestrador / contrato | Micro **120 s** | Ciclo a cada 120 s; RISE_FALL de 120 s |
| Resolução direcional | TCN + meta GBDT | `dl_direction` da TCN; meta refina score / D-SQUEEZE (opcional) |
| Execução contínua | Micro **120 s** | Boleta CALL/PUT na virada do bloco quando há sinal válido |

Com `granularity: 600` e `training_history_bars: 23328`:

| Conceito | Barras | Tempo aproximado |
|----------|--------|------------------|
| Histórico de treino | 23328 | ~162 dias |
| Lookback | 72 | **~12 h** de contexto por sequência (@ 600 s) |
| Validação holdout | ~15% | proporcional ao split |

---

## 4. Camadas de decisão (qualidade)

Ordem lógica de uma entrada:

1. **Fase** — todos os modelos com treino da sessão concluído.
2. **Dados** — histórico suficiente (`gate_reason=data`).
3. **Treinamento** — modelo do símbolo treinado na sessão (`gate_reason=training`).
4. **Predição DL** — inferência Triton concorrente (timeout **0,50 s**, fail-closed); `raw_prob`/`calibrated_prob` e indicadores calculados.
5. **Bundle cross-symbol** — `prepare_meta_classifier_cross_symbol_bundle` coleta telemetria micro em paralelo e anexa spreads cross-symbol.
6. **Stacking tabular** — `MetaClassifierClient` envia probabilidade TCN + vetor **43D** ao `aether-meta-classifier`; retorna `predicted_payoff_edge` (opcional para execução).
7. **Calibração** — `dl_calibration_tolerance`: zona neutra ON (`neutral_half_width: 0.04`); override TCN macro em raw extremos.
8. **Resolução direcional** — `execution_direction_resolver` + `execution_direction_checks` + `meta_payoff_regression`: edge positivo preserva TCN; edge `< -0.15` em squeeze rebaixa `trade_score=0.52` (`[D-SQUEEZE]`); `ensure_direction_margin` expõe margem corrigida.
9. **Gate de qualidade** — dual soft TCN+meta + HARD microestrutura; starvation a partir de **6** skips; stubs sniper não vetam.
10. **Z-Score meta** — `attach_payoff_edge_zscore_metrics` anexa `meta_payoff_edge_zscore` / `edge_zscore` para ranking e gate.
11. **Deploy** — `deploy_ok=false` bloqueia execução; mini-deploy de treino usa `force_local=True` (modelo em memória).
12. **Seleção** — `market_decision_score` multiplicativo (TCN × fator Z-Score); redirect inter-símbolo quando âncora degradada.
13. **Risco** — Kelly em EXPLORE (`fraction: 0.08`, teto 3,5%); Soft Recovery amortizado em RECOVER (`amort_cycles` 2–5, teto `max_safe_stake_pct`); stop win por sessão (2,60% composto ou $10 fixo se banca < $100). Stop loss interno desativado.

Bloqueio absoluto para falhas técnicas (`data`, `predict_error`, `training`, `deploy_ok=false`, Triton) e reconciliação pendente. Vetoes HARD de microestrutura bloqueiam independentemente do soft. Não há vetos táticos autônomos de quality guard soft, cooldown pós-LOSS, blackout de broker ou stubs sniper.

Perfil em `config/settings.json` (settings atuais):

| Parâmetro | Valor | Função |
|-----------|-------|--------|
| `calibration.method` | auto | Platt, isotonic ou temperatura+Platt no holdout |
| `calibration.neutral_half_width` | 0.04 | Zona neutra ON; banda efetiva `[0.46, 0.54]` |
| `confidence_call_threshold` | 0.54 | Threshold CALL |
| `confidence_put_threshold` | 0.46 | Threshold PUT |
| `dynamic_threshold.enabled` | false | Thresholds flutuantes por volatilidade **desligados** |
| `min_val_accuracy` | 0.60 | Piso de acurácia de validação (treino/deploy) |
| `min_validation_accuracy_gate` | 0.63 | Veto HARD de microestrutura em runtime |
| `min_edge_execute` | 0.0 | Edge base (advisory) |
| `label_mode` | `spot_forward` | Rotulagem spot-forward (padrão); `ma_trend` / `triple_barrier` via config |
| `label_vol_window_bars` | 15 | Janela de σ para largura de barreira (tunável por símbolo) |
| `label_vol_multiplier` | 1.0 | Multiplicador da barreira de volatilidade |
| `indicator_gating.enabled` | true | Gate de indicadores |
| `indicators.*` | (settings) | Periodos/multiplicadores/thresholds 100% JSON; mudar periodos exige retreino TCN |
| `quality_gate.starvation.*` / `progressive_conviction.*` / `recovery_relax.*` | (settings) | Inanição, convicção progressiva e relax de recovery (SSOT JSON) |
| `soft_recovery.*` / `recovery_state.*` / `kelly.*` runtime | (settings) | Soft recovery completo, extremos de recovery e turbo/compressão Kelly |
| `loss_protection.disconnect.*` / `market_rank.composite.*` / `edge_zscore.*` | (settings) | Disconnect, ranking composto e janela Z operacional |
| `live_signal_metrics.*` / `meta_payoff_veto.*` / `calibration.*` bounds | (settings) | Live ECE/drift, veto meta e temperatura/trust/TCN override |
| `orchestrator.*_timeout_*` / `infra.meta_classifier.*` / `api_config.stream_reconnect.*` | (settings) | Timing/infra fail-closed (sem defaults mágicos no Python) |
| `indicator_gating.adx_min` | 0.20 | Piso ADX |
| `indicator_gating.vol_ratio_min` | 0.65 | Piso vol_ratio |
| `indicator_gating.veto_on_noise` | false | Sem veto por faixa Hurst de ruído |
| `quality_gate.min_adx_threshold` | 0.20 | HARD ADX starvation |
| `quality_gate.regular.min_direction_margin` | 0.03 | Hard gate de margem (regime regular) |
| `quality_gate.min_direction_margin` | 0.03 | Hard gate de margem (recovery) |
| `quality_gate.mandatory_min_trade_score` | 0.52 | Piso de score em mandatory pick |
| `mandatory_trade_each_cycle` | false | Sem ordem obrigatória a cada ciclo |
| `force_trade_every_cycle` | false | Sem síntese forçada |
| `price_zone.*` | enabled | Zona BB/Keltner + tendência + TCN (AND) |
| `require_meta_for_execution` | false | Meta **opcional** para execução |
| `bb_width_adaptive_squeeze.enabled` | false | Squeeze adaptativo desligado |
| `loss_protection.min_direction_margin` | 0.03 | Piso de margem na proteção contra loss |
| `loss_protection.max_edge_without_margin` | 999.0 | Cap edge sem margem |
| `loss_protection.max_zscore_without_margin` | 999.0 | Cap Z sem margem |
| `risk_management.kelly.max_stake_pct` | 0.035 | Teto Kelly efetivo |
| `risk_management.kelly.max_bankroll_stake_fraction` | 0.035 | Teto de fração de banca alinhado ao Kelly |
| `risk_management.kelly.fraction` | 0.08 | Fração Kelly base (EXPLORE; compressão 40% fora de recovery) |
| `infra.triton.require_for_execution` | true | Timeout Triton falha fechado (sem fallback eager local) |
| `infra.triton.infer_timeout_seconds` | 0.50 | Timeout gRPC de inferência |
| `consensus_penalty_enabled` | false | Consensus Entropy Penalty **desligado** |
| `orchestrator.settlement_tolerance_window_seconds` | 90 | Janela de settlement |
| `orchestrator.watchdog_stale_tick_seconds` | 25 | Watchdog de inanição |

Cover de recovery fraciona o passivo em `amort_cycles ∈ [2,5]` em vez de liquidar `pending/payout` em um trade. Turbo de stake (Z≥1.5) **nunca** ultrapassa `max_safe_stake_cap` pós-multiplicador. Z-Score de edge é bufferizado **por símbolo**. Boot emite `CFG_RISK` via `validate_engine_risk_config` / `RiskPolicy`. Labels train/deploy compartilham `LabelSpec` (`horizon` + `smooth_bars`); treino meta usa proxy de retorno **passado**, não forward.

#### Defaults `risk_management.soft_recovery` (`soft_recovery_policy`)

| Parâmetro | Padrão | Função |
|-----------|--------|--------|
| `enabled` | true | Ativa soft recovery paramétrico |
| `max_safe_stake_cap` | 4.20 | Teto absoluto só em micro-banca a partir de \(N\ge 4\) |
| `max_safe_stake_pct` | 0.035 | Teto percentual de banca em conta ≥ $100 (alinhado ao Kelly) |
| `amort_cycles_min` / `amort_cycles_max` | 2 / 5 | Janela de amortização do cover `pending/payout` |
| `coing_redirect_drawdown_threshold` | 15.00 | Limiar absoluto (USD) do Consensus Cointegration Redirect |
| Passo fixo | n∈{3,4} → U×1.15 | Sem escalada exponencial nesses níveis |
| Hard floor | 5% se banca &lt; $100 | Teto defensivo de stake em micro-banca |
| Micro-residual Z floor | −0.60 | Relaxa veto Z sob passivo residual |
| Waiver GBDT | 6 skips | Antecipa mitigação do veto tabular |

---

## 5. Ranking multiplicativo TCN × Z-Score

O ranking de execução (`execution_market_rank`) substitui a soma linear TCN + meta por uma **ponderação multiplicativa** que prioriza convicção estatística validada pelo LightGBM:

```
zscore_rank_factor(z) = max(0.1, 1.0 + z)
market_decision_score ≈ tcn_score × zscore_rank_factor(meta_payoff_edge_zscore)
```

| Cenário | TCN | Z-Score meta | Efeito |
|---------|-----|--------------|--------|
| Âncora degradada | 0,75 | -1,50 | Score comprimido (~0,075 × TCN base) |
| Par forte validado | 0,68 | +1,20 | Score ampliado (~2,2 × TCN base) |
| Resultado esperado | — | — | Par forte ranqueia acima mesmo com TCN menor |

Telemetria de direção por ciclo:

| Linha | Conteúdo |
|-------|----------|
| `DIR_SEL` | Direção executada, símbolo, edge; `dl=... inv` apenas se houver inversão real |
| `EXEC_SEL` | TCN, edge contínuo e Z-Score (`Z=±x.xx`) — sem rótulo de expectativa |
| `IND` | Snapshot de indicadores + `MARGIN`, `NEUTRAL` (`neutral_clamp`/`calibrated`), `META_VETO` (`none`/`soft`/`hard`) |

O buffer histórico **não** avança durante recovery ou com `linear_losses > 0`, evitando contaminação estatística em soft recovery.

### 5.1 Redirect inter-símbolo (modo contínuo)

Com `mandatory_trade_each_cycle: true`, `try_inter_symbol_zscore_redirect` evita inanição operacional sem forçar entrada degradada:

| Gatilho | Limiar |
|---------|--------|
| Âncora degradada | `meta_payoff_edge_zscore < -0.50` |
| Par alternativo forte | `meta_payoff_edge_zscore > +0.50` |

O fluxo inteiro do ciclo desvia para o par com maior probabilidade matemática de ganho no milissegundo de entrada.

---

## 6. Resolução direcional com edge contínuo e downgrade D-SQUEEZE

`execution_direction_resolver.resolve_execution_direction` delega o refinamento a `meta_payoff_regression.apply_meta_regression_edge`; pré-checagens em `execution_direction_checks`:

| Etapa | Regra |
|-------|-------|
| `dl_direction` | TCN: `P(CALL) > pivot` → CALL, caso contrário PUT |
| Pré-checagens | Rejeita ciclo apenas por starvation de microestrutura; limpa `neutral_clamp`; hooks sniper (stubs) |
| Payoff GBDT | `predicted_payoff_edge` do container LightGBM (porta 8005); opcional para execução |
| Edge positivo | `predicted_payoff_edge > 0.0` → `exec_direction = dl_direction`; `trade_score` orgânico da TCN |
| Downgrade squeeze | `predicted_payoff_edge < -0.15` **e** (`bb_width < 0.06` **ou** `micro_tick_acceleration < 0`) → `trade_score=0.52`; `meta_squeeze_downgrade=true`; log `[D-SQUEEZE]` |
| Edge negativo leve | Mantém `dl_direction` e score orgânico |
| `direction_margin` | `abs(P(lado_escolhido) − 0.50)` — distância ao neutro; recalculada por `ensure_direction_margin` no retorno do resolver |
| `direction_inverted` | `False` no fluxo de regressão |

**Justificativa do score 0.52 em squeeze**: em canais Bollinger esmagados, o rebaixamento marca `meta_squeeze_downgrade` e, **fora de recovery financeira** (`pending_total == 0`), comprime a stake ao piso de $1.00 da Deriv. Com `pending_total > 0`, o piso D-SQUEEZE é **revogado** (`d_squeeze_floor_waived_for_recovery`) para preservar a stake soft D'Alembert e cobrir o passivo; o ranking de recovery tambem penaliza simbolos em squeeze quando ha peer elegivel. O adaptativo `bb_width_adaptive_squeeze` permanece **desabilitado** nos settings atuais.

`execution_direction_cross_corr` e `execution_volatility_booster` permanecem como telemetria consultiva. `execution_sniper_gates.apply_hurst_noise_veto` e `apply_bb_squeeze_requirement` são stubs que retornam `False`.

---

## 7. Kelly e Consensus Entropy Penalty (base)

Quando a ordem final (`order_direction`) diverge da maioria dos votos técnicos (`call_votes`/`put_votes`):

| Etapa | Definição |
|-------|-----------|
| Concordância | `agreement = votos_alinhados / (call_votes + put_votes)` |
| Divergência | `divergence = 1 - agreement` |
| Penalidade convexa | `penalty = divergence^exponent × (w_di·|di_opp| + w_cmo·|cmo_opp| + w_rsi·|rsi_opp|)` |
| Retenção Kelly | `retention_raw = max(floor, 1 - min(max_cut, penalty))` |
| Stake Kelly | `f*_efetivo = f* × retention` |

Em baixo consenso (`retention_raw ≤ consensus_min_retention`, padrão 0,50), a stake é forçada ao piso mínimo da API ($1,00), protegendo contra ruído do CSPRNG quando DL e indicadores clássicos discordam.

**Settings atuais:** `consensus_penalty_enabled: false` — a penalidade permanece no código, mas **não** altera o Kelly base em produção com a config canônica.

**Modo contínuo (quando habilitado):** essa penalidade opera sobre o Kelly base mesmo em recovery. A convergência adaptativa (seção 8.2) evita que a penalidade asfixie a recuperação financeira.

> **Gates defensivos legados removidos / stubs.** O Gate Assimétrico de Proteção, o Micro Noise Gate, o Filtro de Exaustão de Barreira Micro e o Veto de Inversão por Convicção DL foram eliminados. Sniper/Hurst/BB retornam `False`. Em modo mandatário, o quality guard soft permanece telemetria quando a falha é fraca; falha **fortemente negativa** do meta (`predicted_payoff_edge < 0` ou Z `< -0.20` em todos os candidatos) pode suspender o cluster, salvo waiver de emergencia. Vetoes HARD de microestrutura (`adx_starvation`, `vol_ratio_starvation`, `val_accuracy_gate`) bloqueiam independentemente.

---

## 8. Recovery, sizing e persistência financeira

Esta seção documenta as diretrizes matemáticas que corrigem a **inanição por sizing desalinhado**: WINs operacionais com micro-stakes que zeravam o contador de perdas sem extinguir o drawdown real da sessão.

### 8.1 Filosofia de Recovery Financeiro Persistente

#### Problema: reset cego vs. realidade financeira

Em execução contínua (`mandatory_trade_each_cycle: true`), o motor pode registrar um **WIN operacional** (P&L positivo no contrato liquidado) enquanto o **saldo acumulado da sessão** (`total_session_profit`) permanece negativo e o **drawdown pendente** (`pending_loss`) ainda carrega valor a recuperar.

O critério legado — resetar `consecutive_losses` a zero após qualquer cluster com P&L ≥ 0 — tratava **resultado operacional isolado** como **recuperação financeira completa**. Isso gerava assimetria negativa:

```
Ciclo 1: LOSS  -$10  →  pending_loss = $10,  consecutive_losses_linear = 1,  D'ALEMBERT
Ciclo 2: WIN   +$3   →  pending_loss = $7,   consecutive_losses_linear = 1   ← retração parcial
Ciclo 3: LOSS  -$12  →  nova perda com memória de recovery preservada
```

O robô voltava ao Kelly fracionário com stakes micro (~$8), enquanto a sessão continuava no vermelho — **inanição por sizing desalinhado**.

#### Definições quantitativas

| Variável | Significado |
|----------|-------------|
| `pending_loss[s]` | Drawdown financeiro pendente por símbolo `s`, acumulado após losses e reduzido por wins via `apply_win_to_pending_loss` |
| `pending_total` | `Σ pending_loss[s]` — critério único de recovery financeiro ativo |
| `total_session_profit` | P&L acumulado real da sessão (soma de todos os contratos liquidados pela API) |
| `consecutive_losses_linear` | Contador de clusters negativos — **memória operacional** de stress no soft recovery |
| `dlambert_unit` (U) | Unidade aditiva capturada na primeira stake Kelly da sessão (ou override de config) |
| `recovery_financially_active` | Verdadeiro iff `pending_total > 0` |

#### Regra de persistência (implementação atual)

O motor **não utiliza reset cego** de `consecutive_losses_linear` baseado em WIN operacional isolado.

| Condição após liquidação | Comportamento |
|--------------------------|---------------|
| `cluster_profit < 0` | `consecutive_losses_linear += 1`; `pending_loss` incrementado |
| `cluster_profit ≥ 0` **e** `pending_total > 0` | WIN absorvido no drawdown; **`consecutive_losses_linear = max(1, n-1)`** (retração do soft D'Alembert) |
| `cluster_profit ≥ 0` **e** `pending_total = 0` | Recovery financeiro extinto; reset de `consecutive_losses_linear` e `last_loss_stake` |

**Persistência de Drawdown:** o robô permanece em estado de Recovery (Soft Recovery / perdas lineares) até que `pending_total` seja **financeiramente zerado** por retornos reais da API — não por um WIN simbólico que não cobre o buraco acumulado.

#### Implicação para gestão de cauda

- O **estado de risco** segue o **passivo financeiro** (`pending_loss`), não a contagem superficial de vitórias.
- Micro-WINs em recovery **amortizam** o drawdown e **retraem** o expoente da curva (`max(1, n-1)`), mas **não encerram** o regime de recovery prematuramente.
- Logs de auditoria: `RISK: WIN operacional`, `RISK: Lucro parcial`, `RISK: Recovery financeiro zerado` — cada um com `pend=$` e `pnl_sess=$`.

---

### 8.2 Waiver de Consensus Penalty em Recovery

#### Problema: penalidade convexa vs. peso financeiro do recovery

O **Consensus Entropy Penalty** comprime `f*` quando a ordem diverge dos votos técnicos. Em recovery, punir a stake por falta de consenso upstream asfixia a recuperação — o soft recovery precisa de peso financeiro real para extinguir o passivo.

#### Bypass de consensus em recovery com passivo pendente

Quando `pending_total > 0`, `risk_stake_calc.py` **ignora** a penalidade de entropia de votação e o piso `stake_min` por divergência de consenso (quando a penalidade estiver habilitada). Com `consensus_penalty_enabled: false`, o Kelly base já não aplica a retenção. O payload segue para soft D'Alembert com tag `D'ALEMBERT`. A stake soft usa `max(U × m(n), pending_total / payout / amort)` limitada pelo teto de banca (`max_safe_stake_cap`):
- **Passo Fixo (n ∈ {3,4})**: `m(n) = 1.15` (Unidade Base + 15%), sem escalada exponencial.
- **Demais níveis**: `m(n) = factor^n` com `factor = min(1 + 1/payout, 2.50)`.
- **Hard Floor** (banca &lt; $100): stake de recovery ≤ **5%** do saldo atual.

#### Condições de waiver absoluto (`retention = 1.0`) quando consensus estiver ligado

1. **Recovery ativo:** `pending_total > 0` **ou** `consecutive_losses_linear > 0`
2. **Qualquer** candidato do cluster em recovery que atenda **uma** das condições:
   - **Votos unânimes alinhados:** `6×0` ou `0×6` na direção da ordem (macro)
   - **Convicção elevada:** `trade_score >= penalty_smoothing_trade_score_min` (padrão **0,68**)

Justificativa: com alinhamento direcional unânime no contexto macro ou convicção alta, o Kelly base não pode ser esmagado pela penalidade de entropia — o soft D'Alembert precisa operar com peso financeiro real em símbolos secundários do cluster (ex.: `R_10`).

---

### 8.3 Sizing Kelly + Soft Recovery (operacional)

#### Filosofia: Kelly na exploração; Soft Recovery na recovery financeira

Com `soft_recovery.enabled: true`, o switch em `calculate_stake_for_manager` usa o regime já modelado:

| Regime | Condição | Sizer | Tag |
|--------|----------|-------|-----|
| **EXPLORE** | `pending_total == 0` e `linear == 0` | Kelly fracionário (`fraction: 0.08`, tetos 3,5%) | `EXPLORE_KELLY` |
| **RECOVER** | `pending_total > 0` ou `linear >= 1` | Soft Recovery amortizado (`amort_cycles` 2–5, teto `max_safe_stake_pct`) | `RECOVER_DAL_Ln` / `D'ALEMBERT` |

#### Soft Recovery (path canônico)

Em recovery ativo, o sizing adota progressão adaptativa (`soft_recovery_policy` + `apply_soft_recovery_stake` / `dlambert_sizing.resolve_dlambert_stake`), limitada por `max_safe_stake_pct` (conta grande) ou `max_safe_stake_cap` (micro) e tetos Kelly.

| Estado | Stake |
|--------|-------|
| Normal | Kelly fracionário + tag `EXPLORE_KELLY` |
| Recovery | Soft amortizado; tag `RECOVER_DAL_Ln` |

| Termo | Significado |
|-------|-------------|
| `U` (`dlambert_unit`) | Unidade de recovery (pode ser override ou capturada na sessão) |
| `consecutive_losses_linear` | Contador de stress / memória operacional |
| `max_safe_stake_pct` | Teto defensivo % banca (default 3,5%) |
| `max_safe_stake_cap` | Teto absoluto só em micro-banca |

#### Retração em WIN parcial

```
WIN parcial (pending_total > 0 após liquidação):
  consecutive_losses_linear = max(1, n - 1)

WIN total (pending_total = 0):
  consecutive_losses_linear = 0               → volta ao Kelly puro
```

O expoente **sobe** uma unidade por LOSS e **desce** uma por WIN parcial, nunca abaixo de 1 enquanto recovery ativo — evitando reset cego sem extinguir o drawdown.

#### Sem circuit breaker

Foram removidos `dlambert_circuit_breaker`, as constantes `MAX_LINEAR_LEVEL`, `MAX_STAKE_U_MULTIPLE`, `MAX_SESSION_DRAWDOWN_U` e o curto-circuito da tag `D'ALEMBERT_CB` em `calculate_stake_for_manager`. O cálculo prossegue livre na thread principal, sem forçar stake a `0.0` nem travar exposição.

#### Exemplo numérico

```
Sessão: U = $8 (unidade soft recovery)

Kelly puro:     n=0 → stake ≈ Kelly fracionário
LOSS #1:        n=1 → stake = max(U × factor^1, cover)
LOSS #2:        n=2 → stake = max(U × factor^2, cover)
LOSS #3:        n=3 → stake = max(U × 1.15, cover)   Passo Fixo
LOSS #4:        n=4 → stake = max(U × 1.15, cover)   Passo Fixo
WIN parcial:    n=max(1,n-1); pending_total > 0
WIN total:      n=0 → volta ao Kelly puro
```

#### Parâmetros de configuração (`risk_management.soft_recovery` / `soft_recovery_policy`)

| Parâmetro | Padrão | Função |
|-----------|--------|--------|
| `enabled` | true | Ativa Soft Recovery Adaptativo em recovery |
| `max_safe_stake_cap` | 4.20 | Teto absoluto em micro-banca a partir de \(N\ge 4\) |
| `amort_cycles_min` / `amort_cycles_max` | 2 / 5 | Janela de amortização do cover `pending/payout` |
| `coing_redirect_drawdown_threshold` | 15.00 | Limiar absoluto (USD) do Consensus Cointegration Redirect |
| Micro-residual | Z floor −0.60; waiver GBDT 6 skips | Baixa intensidade: fecha válvula de cointegração e relaxa veto Z |

O bloco legado `risk_management.dlambert` foi removido da configuração canônica; o motor ainda aceita `dlambert_enabled` apenas como fallback de compatibilidade. A política canônica vive em `soft_recovery_policy.py`.

#### Seleção de símbolo e Hurst em recovery

Complementar à curva soft D'Alembert:

- Rotação de símbolo após loss linear (`symbol_loss_rotation_cycles`); sem bônus fixo em `R_10`
- Filtro de loss-protection com `min_direction_margin: 0.03` (caps edge/Z 999)
- Trava Hurst N2+ (`recovery_hurst_gate`) — prioriza candidatos persistentes; N3+ sem Hurst bloqueia escalada
- Teto de stake comprimido em linear ≥2 (2,5%) e ≥3 (2,0%)

---

### 8.4 Barreira de persistência pós-reset linear

Quando um cluster encerra com reset linear D'Alembert (`_linear_reset_occurred`), o motor executa uma sequência atômica antes de permitir novo ciclo de inferência:

1. `_finalize_linear_reset_risk_state` — zera `consecutive_losses_linear`, `last_loss_stake` e limpa entradas zeradas de `pending_loss`.
2. `_persist_session_state_snapshot` — espelha saldo em `StateManager` e persiste `data/session_state.json`.
3. `_persist_full_state_unlocked` — bundle Redis sem reentrância no lock.
4. Yield cooperativo de **0,1 s** — libera o event loop antes do próximo ciclo DL.

Durante a barreira, `session_persistence_write_active` impede que `trading_cycle_entry` execute `execute_cluster` concorrentemente. O lock central (`StateManager._state_lock`) garante que leituras de stake e escrita de liquidação não se sobreponham — eliminando deadlocks térmicos silenciosos observados na transição entre clusters (ex.: C0006 → inferência subsequente).

---

### 8.5 Side equilibrium — leis dos pequenos e grandes números

Módulos: `domain/analytics/side_equilibrium.py`, `side_equilibrium_gate.py`, `side_equilibrium_store.py`. Config: `orchestrator.execution.side_equilibrium`.

| Regime | Janela | `n_min` | Ação típica |
|--------|--------|---------|-------------|
| **Small-N** (lei dos pequenos números) | 12 trades | **2** | `hard_skip` se WR do lado &lt; `wr_floor_small` (0.40) ou frequência do lado ≥ `freq_bias_max_small` (0.70) |
| **Large-N** (lei dos grandes números) | 100 trades | 40 | `soft_penalty`: `kelly_mult_soft` (0.55) e `margin_boost_soft` (0.03) se WR &lt; `wr_floor_large` (0.48) ou bias ≥ 0.65 |
| Amostra insuficiente | `n_side &lt; n_min` ou `total &lt; n_min` | — | `pass` (log `SIDE_EQ … action=pass`) |

Telemetria: `SIDE_EQ | SYMBOL SIDE | call=W/N put=W/N | bias=… wr=… | action=…` (dedupe por ciclo/símbolo/lado). Outcomes em `process_contract_outcome`. Hard-skip no proposto **tenta o oposto** (`SIDE_EQ_FLIP`); com 2 PUT LOSS (wr=0, bias=1) flipa para CALL mantendo Soft Recovery/Kelly. Soft penalty escala `kelly_fraction_scale` (path Kelly EXPLORE).

---

## 9. Execução

| Flag | Efeito |
|------|--------|
| `mandatory_trade_each_cycle: false` | Entrada seletiva: só com `price_zone` BUY/SELL alinhada à direção |
| `mandatory_trade_each_cycle: true` | Esteira contínua (legado; desligado nos settings atuais) |
| `require_meta_for_execution: false` | Meta opcional; Triton permanece fail-closed |
| `include_anchor_trades` | Inclui âncora nas ordens do cluster |
| `diversify_after_loss_margin` | Prefere símbolo alternativo quando scores são próximos |

Logs: `ord=` (ordem enviada) sempre igual a `dl=` (direção prevista pelo DL), pois não há mais inversão.

---

## 10. Risco e stop win por sessão

| Mecanismo | Papel |
|-----------|-------|
| Kelly fracionário | Sizing EXPLORE com win rate dinâmico; compressão 40% fora de recovery (`fraction: 0.08`) |
| Target Proximity Damping | Amortecimento linear da stake Kelly conforme `pnl_sessao` se aproxima de `target_win` (piso 0.40×) |
| Consensus Entropy Penalty | Presente no código; **desligado** nos settings atuais (seção 7) |
| Penalty Smoothing | Convergência adaptativa em recovery quando consensus estiver ligado (seção 8.2) |
| Recovery financeiro persistente | Estado de risco atrelado a `pending_total` (seção 8.1) |
| Sizing | Kelly EXPLORE + Soft Recovery RECOVER (seção 8.3) |
| Side equilibrium (LLN) | Small-N / large-N CALL/PUT (seção 8.4) |
| Stop win por sessão ativa | Banca ≥ $100: `target_win = session_start_balance × compounding_rate_daily` (padrão 2,60%); banca < $100: stop win fixo `$10`; fast-path anti-deadlock |
| Stop loss | Desativado — sem reset por relógio nem disjuntor de perda diária |

### 10.1 Juros compostos e controle operacional

A meta segue a planilha de gerenciamento de juros compostos (`compounding_rate_daily: 0.026`), com override de micro-banca:

| Evento | Comportamento |
|--------|---------------|
| Boot do processo | Captura saldo Deriv (ou `session_start_balance` em settings) como `session_start_balance` |
| Meta calculada (banca ≥ $100) | `target_win = session_start_balance × 0,026` (arredondada para baixo em centavos) |
| Meta calculada (banca < $100) | `target_win = small_account_stop_win` (padrão `$10`), mesmo com compounding ativo |
| Durante a sessão | `pnl_sessao = current_balance - session_start_balance` |
| Meta atingida | Fast-path: `clear_current_session_redis_keys` → `cancel_settlement_queue_fast` → `graceful_shutdown(fast_path=True)` |
| Ciclo pós-liquidação incompleto | Retry com breath; após 2 falhas consecutivas → `emergency_save_session_state` + `recover_post_settlement_loop_transparently` |
| Restart manual | Nova sessão independente com novo saldo e nova meta |
| Mesmo dia civil | Múltiplas sessões isoladas permitidas — sem virada UTC/meia-noite |

Parâmetros em `risk_management` / `risk_management.params`:

| Chave | Padrão | Função |
|-------|--------|--------|
| `compounding_enabled` | `true` | Ativa meta composta por sessão |
| `compounding_rate_daily` | `0.026` | Taxa de juros (2,60% sobre banca inicial ≥ $100) |
| `session_start_balance` | `null` | Override manual da banca inicial (senão usa saldo Deriv) |
| `small_account_threshold` | `100.0` | Limiar abaixo do qual o stop win é fixo |
| `small_account_stop_win` | `10.0` | Stop win fixo em dólares para micro-banca |
| `duration` | `120` | Duração do contrato RISE_FALL (s) |

Com `compounding_enabled: false`, o motor recorre ao alvo legado (`small_account_stop_win` / `large_account_stop_win_pct`).

### 10.2 Sizing defensivo de proximidade de alvo

Evita superexposição quando a sessão já capturou a maior parte do stop win:

1. **Kelly base** — `resolve_effective_kelly_fraction` usa `kelly.fraction` de config (**0,08**), com target proximity em regime EXPLORE (consensus off).
2. **Amortecimento dinâmico** — após o Kelly bruto, `apply_kelly_target_proximity_damping` multiplica a stake por `0.40 + 0.60 × remaining_target_pct`.
3. **Exemplo** — meta $101.20, Kelly bruto $45.56: com `pnl_sessao = 0` permanece $45.56×1.0 (já atenuado pela fração base); com 90% da meta (`pnl ≈ $91.08`) o fator cai para 0.46 (~$20.96).

Fora de recovery, este amortecimento define o `Kelly_base`. Em RECOVER com Soft Recovery, a stake amortiza `pending` sob `max_safe_stake_pct`.

Log de bootstrap (banca ≥ $100): `SESSAO INICIADA | Alvo de 2,60%: $XX.XX | Stop Loss: DESATIVADO`.

Log de bootstrap (banca < $100): `SESSAO INICIADA | Alvo fixo micro-banca: $10.00 | Stop Loss: DESATIVADO`.

---

## 12. Normalização Adaptativa de Volatilidade & Drift Bias Lock

### 12.1 Normalização Adaptativa e Clipping de Volatilidade
Para evitar degradação de sinal e problemas OOD (Out-of-Distribution) no LightGBM em períodos de estouro dinâmico de volatilidade, as features de dispersão temporal `bb_width`, `atr_norm` e o bloco de micro-volatilidade (`micro_bid_ask_spread_momentum_zscore`, `volatility_shadow_ratio_zscore`) são normalizadas com base nas estatísticas das últimas 1024 velas macro do TimescaleDB:
\[X_{\text{zscore}} = \frac{X - \mu_{1024}}{\sigma_{1024} + 1e-12}\]
Se o valor ultrapassar o teto crítico de ±3.0, aplica-se um clipping estrito via:
\[X_{\text{final}} = \text{clip}(X_{\text{zscore}}, -3.0, 3.0)\]
O payload HTTP do meta (`META_FEATURE_DIM = 43`) espelha rigidamente essa saturação antes do envio ao container `aether-meta-classifier` (porta 8005).

### 12.2 Invariante de Drift Proibido (Drift Bias Lock)
Com universo single-symbol (`R_10`), o lock contra drift natural de par Bull/Bear permanece no codigo como no-op (`hedge_peer(R_10)` retorna `None`). Em configuracoes legadas com par Drift (`RDBULL`/`RDBEAR`), continua vedada a emissao contra o drift sob expansao hiperbolica de volatilidade (\(Z_{\text{vol}} \ge 2.0\)):
- **PUT** contra tendencia de alta no indice Bull
- **CALL** contra tendencia de baixa no indice Bear

### 12.3 Salvaguarda de Micro-Banca e Válvula Adaptativa de Cointegração
Para estender a sobrevida do patrimônio em contas de micro-capital ($100 ou menos), a unidade base \(U\) do soft D'Alembert e a válvula de cointegração operam assim:

**Unidade base dinâmica**
\[
U = 0{,}01 \cdot B \quad (B \le 250),\qquad U = 0{,}0015 \cdot B \quad (B > 250)
\]
Para \(B = 100\), \(U = \$1{,}00\).

**Trava de cauda (`max_safe_stake_cap`)** — a progressão geométrica soft é achatada a partir do nível linear \(N \ge 4\):
\[
S_{\max}(B, N) =
\begin{cases}
4{,}2 \cdot U & \text{se } B \le 250 \land N \ge 4 \\
p(N) \cdot B & \text{caso contrário}
\end{cases}
\]
com \(p(0)=0{,}035\), \(p(N\!\ge\!2)=0{,}025\), \(p(N\!\ge\!3)=0{,}020\). Em \(B=100\) e \(N\ge 4\): \(S_{\max}=\$4{,}20\), o que preserva fôlego para \(>24\) ciclos adversários sequenciais sob stake unitária.

**Consensus Cointegration Redirect** — quando o passivo da sessão ultrapassa o limiar configurado (`coing_redirect_drawdown_threshold: 15.00`) ou 15% do capital vivo:
\[
\mathbb{1}_{\text{redirect}} = \mathbf{1}\!\left[B \le 250 \land P > 0{,}15\cdot B\right]
\]
com \(P=\sum_s \text{pending\_loss}[s]\) (\$15 em banca de \$100). Sob \(\mathbb{1}_{\text{redirect}}=1\), ordens isoladas em ativos de maior ruído são suspensas e o payload de soft recovery é desviado ao simbolo operacional `R_10` (ou, em testes multi-symbol, ao candidato em `DRIFT_PAIR_SYMBOLS`) que maximiza
\[
\text{score}(s) = Z_{\text{edge}}(s) - H_2\!\big(p_s\big),\qquad Z_{\text{edge}}(s) > 0
\]
onde \(H_2\) é a entropia de Shannon binária e \(p_s\) a probabilidade calibrada do símbolo. O candidato com maior \(\text{score}\) absorve a execução do ciclo.

### 12.4 Válvula Dinâmica de Payoff GBDT
Em estados extremos de inanição operacional (skips >= 30), o veto do classificador tabular é mitigado de forma cooperativa para evitar o congelamento permanente do soft recovery:
- Se o contador `skipped_cycles_counter` atingir ou exceder 30 ciclos consecutivos pulados por qualidade, e houver concordância unânime de votos técnicos (6x0 no domínio), o veto do payoff do LightGBM ('Meta Payoff < min') é ignorado, permitindo que a força bruta limpe o passivo financeiro.
- Sob **Micro Passivo Residual** (banca ≤ $250, `pending_total` ≤ $5 e < 5% da banca), o piso do veto `meta_payoff_negative_zscore_veto` relaxa de \(Z \ge -0.20\) para \(Z \ge -0.60\), a válvula de cointegração permanece fechada e o waiver GBDT antecipa para 6 skips — evitando loops `EXEC_EMPTY` em passivo centavado. Após `EXEC_EMPTY` em recovery, o orquestrador alinha o cooldown à fronteira de assinatura de **120 s** (sem retries fragmentados de 8s).

---

## 11. Referências internas

- [arquitetura.md](arquitetura.md)
- [structure.md](structure.md) — inventário completo dos módulos Python em `app/src/` (~226)
- [README.md](../README.md)
- [CHANGELOG.md](CHANGELOG.md)
