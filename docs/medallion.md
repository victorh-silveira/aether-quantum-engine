# Metodologia quantitativa

O Aether Quantum Engine herda a postura **Medallion** no sentido operacional: o mercado é um **sistema de sinais ruidosos**, não uma narrativa macro discricionária. A implementação concentra-se no índice sintético **`1HZ75V`** (Volatility 75 (1s)) com **Deep Learning** (TCN) e classificação binária Rise/Fall.

Para arquitetura de código, ver [`arquitetura.md`](arquitetura.md).

Doutrina do copiloto LLM/Cursor (9 livros → constraints de engenharia): [`llm-trading-doctrine.md`](llm-trading-doctrine.md).

---

## 1. Princípios

| Princípio | No motor atual |
|-----------|----------------|
| Sinais, não histórias | Direção CALL/PUT estritamente pela TCN + fusão EV (`fusion_p_eff`) |
| Horizonte e Timeframe | Contexto DL macro **86400 s** (D1 / 365 barras); micro/MINI OHLC **300 s** (M5 / 500 barras); contrato RISE_FALL **5 m** (ops fixo); label `quantum_multi_barrier` (horizonte N=1 vela M5); proporção multi-timeframe **1:288** (300:86400) |
| Acoplamento temporal | Inferências e rotações seguem `signature_boundary_seconds` (**300 s**) com ciclo em **120 s** |
| Esteira contínua | `mandatory_trade_each_cycle: false` (sem vetos arbitrários; fusão EV + signal_skip 1.1 + anti-loss microestrutura M5) |
| Force trade | `force_trade_every_cycle: false` — sem síntese forçada de candidato |
| Modelo pronto antes de operar | `FASE TREINO` suspende ordens até treino da sessão |
| Fail-closed seletivo | Meta **opcional** nos settings atuais; TCN eager/CUDA local no host |
| Feedback real | Win rate live integrado; loss-classifier e meta-classifier treinados online pós-settle via `/v1/learn` |
| Defesa contra ruído | Anti-loss com microestrutura balanceada M5: EMA slope 9/21, RSI momentum e trava de pânico bilateral Z-Score |
| Persistência financeira | Recovery atrelado a `pending_loss`, amortização suave em 2 a 3 ciclos (`cover_multiple` **1.10**) |
| Sizing Single-Strike | Kelly Single-Strike projetado para atingir **4,31% da banca em tacada única M5** com cap de **5,0%** |
| Side equilibrium (LLN) | `sample_size_policy` + `side_equilibrium`: runtime aplica soft Kelly sem flip forçado de direção |
| Meta por sessão ativa | Stop win de **4,31%** composto — encerra a sessão com sucesso (`EXEC_PAUSE`) |
| Sem disjuntor de perda | Stop loss interno desativado por política do operador |
| Isolamento de estado | `asyncio.Lock` serializa inferência, liquidação e persistência atômica |
| Calibração e Zona Neutra | Zona neutra gera estritamente `SKIP:NEUTRAL_ZONE` com `execute=False` |
| Settlement resiliente | Fila Redis `settlement:queue:priority`; tolerância **600 s**; reconciliação passiva |

---

## 1.1 Lei dos Grandes Numeros (operacional)

Inspirado em Mlodinow (*O Andar do Bebado*, caps. 3–4): amostras pequenas sao ruido; a media so converge com volume. O motor trata isso como politica SSOT em `orchestrator.execution.sample_size_policy`.

| Ideia do livro | No codigo |
|----------------|-----------|
| Lei dos Grandes Numeros (Bernoulli) | `evidence_n_min=12` / `large_n_min=32` antes de confiar em WR live, ECE e soft sizing por underperformance |
| Vies dos Pequenos Numeros (Tversky/Kahneman) | `n_min_small=8`: 2–3 losses nao geram SKIP de direcao nem toxic label |
| Falacia do apostador | Recovery nao escala por “autocorrecao”; calib drift soft exige `calib_soft_min_n=15` |
| Mao quente | `explore_stake_scale` e shrink bayesiano diluem streaks curtas em direcao ao prior |
| Diluicao, nao magia | `empirical_rate_shrink` e `sample_reliability = n/(n+half_life)` |

Modulo: `app/src/domain/analytics/sample_size_policy.py`. Integrado em SIDE_EQ, Kelly bayesiano, `apply_kelly_fraction_scale` (EXPLORE) e `apply_live_calib_drift_soft`.

A doutrina LLM estende o mesmo raciocinio aos demais livros (Taleb, Duke, Douglas, Murphy, LTCM, etc.) em [`llm-trading-doctrine.md`](llm-trading-doctrine.md).

---

## 2. Universo Drift e perfil de qualidade

### 2.1 Universo Drift

Índices sintéticos correlacionados no eixo de barreiras. Cada símbolo tem modelo DL independente com **34 features** e volatilidade calibrada ao alvo do índice.

| Símbolo | Papel típico |
|---------|----------------|
| `1HZ75V` | Universo operacional unico; ancora e unico simbolo de treino/execucao |

Operação: contratos **RISE_FALL** de **5 m** (CALL = alta no período do contrato, PUT = queda). Ciclo **120 s**; OHLC micro/MINI em **300 s** (M5; label TCN **N=1 vela M5**; alinhado ao contrato ops 5 min).

### 2.2 Telemetria de Volatilidade, Exaustão e Fluxo Micro

Indicadores micro de **300 s** (M5) (RSI, `vol_ratio`, Keltner, `bb_width`, aceleração de ticks, shadow de volatilidade e momentum de spread) alimentam o container `aether-meta-classifier` (porta **8005**) via vetor **43D**, indexados na resolução amostral micro do TimescaleDB. O `LGBMRegressor` (huber) estima `predicted_payoff_edge` contínuo; o resolver preserva score orgânico da TCN quando o edge é positivo e aciona downgrade D-SQUEEZE quando o edge colapsa em microestrutura. Nos settings atuais, meta é **opcional** para execução.

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
| `micro_tick_acceleration` | Aceleração estocástica de ticks no bloco micro corrente (300 s / M5) |
| `keltner_deviation_ratio` | Distância fracionária do último tick ao centro do canal Keltner micro |
| `micro_bid_ask_spread_momentum` | Taxa de variação de ticks aglutinados por sub-janelas de 5 segundos no bloco micro corrente |
| `micro_bid_ask_spread_momentum_zscore` | Z-Score adaptativo histórico de 1024 períodos da variação de ticks, clipado a ±3.0 |
| `volatility_shadow_ratio` | Razão entre a soma dos pavios (superior + inferior) da barra micro atual e a amplitude do desvio padrão do Keltner (ATR) |
| `volatility_shadow_ratio_zscore` | Z-Score adaptativo histórico de 1024 períodos da razão de pavios, clipado a ±3.0 |

Indicadores macro (Hurst, ADX, bandas) permanecem em `metrics["indicators"]` / `feature_vector` (34D TCN) como telemetria analitica e insumo do stacking — sem veto HARD de microestrutura no pipeline de execucao (escopo 1).

### 2.5 Perfil de qualidade atualizado

| Camada | Comportamento |
|--------|---------------|
| Bloqueio técnico | `data`, `predict_error`, `training`, `deploy_ok=false` |
| Calibração | Zona neutra **off** (`neutral_half_width: 0.0`); thresholds **0.62/0.38**; override TCN macro se raw&gt;0.65 ou &lt;0.35 |
| Veto cruzado TCN-GBDT | Soft comprime score; hard com shadow; soft não hard-blocka o resolve |
| Classificação macro | TCN processa lookback **30** em barras diárias **86400 s** (`[1, 30, 34]`); define direção (`dl_direction`) |
| Stacking tabular | Meta-regressor LightGBM (micro **300 s**) sobre vetor **43D** + probabilidade TCN; saída `predicted_payoff_edge`; meta **opcional** |
| Z-Score de payoff | `payoff_edge_zscore`: janela adaptativa 15–45; classificação estatística do micro-edge |
| Scoring de ranking | `market_decision_score = tcn × max(0.1, 1 + z)` |
| Margem direcional | `direction_margin = abs(P(lado) − 0.50)`; thresholds adaptativos |
| Anti-Loss M5 | Microestrutura estrita: EMA slope 9/21 em barras de 5m, RSI momentum e confirmação de 3 barras |
| Rotulagem | SSOT `quantum_multi_barrier` (barreiras assimetricas + Expiry; alt. `triple_barrier`) |
| Gerenciamento de risco | Kelly Single-Strike 4.31% (`kelly.fraction: 0.08`, cap 5.0%); Soft Recovery amortização 2 a 3 ciclos |

---

## 3. Blindagem multi-timeframe

**Invariante 1:288:** o relógio operacional micro (`data_handler.micro_granularity` = **300 s**) e o contexto macro DL (`data_handler.granularity` = **86400 s**) mantêm proporção **1:288**. Cada bloco diário cobre duzentas e oitenta e oito barras M5. A assinatura `m5b:{boundary};m5:{sym}@{epoch};m15:...` e `seconds_until_next_signature_boundary` ancoram a invalidação de cache na cadência da barra M5 (**300 s**). Contrato Deriv **5 m** (ops fixo); label TCN **N=1** vela M5 (`quantum_multi_barrier`).

| Camada | Timeframe | Papel |
|--------|-----------|-------|
| Deep Learning / TCN | Micro **300 s** / macro **86400 s** (lookback **30**) | Tensor `[1, 30, 34]` no contexto macro D1; proporção 1:288 |
| Meta-regressor GBDT | Micro **300 s** | Regressão tabular **43D**; edge contínuo via `/v2/predict_meta` |
| Orquestrador / contrato | Ciclo **120 s** / RISE_FALL **5 m** | Settle ops em T+5 min; label TCN em N=1 vela M5 |
| Resolução direcional | TCN + fusão EV + anti-loss | Ponderação e filtros de momentum em barras de 5m |
| Execução contínua | Ciclo **120 s** (boundary **300 s**) | Boleta CALL/PUT na cadência M5 quando há sinal válido |

Com `lookback: 30`, `micro_granularity: 300` e `training_history_bars: 365` (1 ano de histórico diário):

| Conceito | Barras | Tempo aproximado |
|----------|--------|------------------|
| Histórico de treino | 365 | 365 dias (D1) |
| Lookback | 30 | 30 dias de contexto macro |
| Validação holdout | ~20% | Split temporal estratificado |

---

## 4. Camadas de decisão (qualidade)

Ordem lógica de uma entrada:

1. **Fase** — todos os modelos com treino da sessão concluído.
2. **Dados** — histórico suficiente (`gate_reason=data`).
3. **Treinamento** — modelo do símbolo treinado na sessão (`gate_reason=training`).
4. **Predição DL** — inferência eager/CUDA local; `raw_prob`/`calibrated_prob` e indicadores calculados.
5. **Bundle cross-symbol** — `prepare_meta_classifier_cross_symbol_bundle` coleta telemetria micro em paralelo e anexa spreads cross-symbol.
6. **Stacking tabular** — `MetaClassifierClient` envia probabilidade TCN + vetor **43D** ao `aether-meta-classifier`; retorna `predicted_payoff_edge` (opcional para execução).
7. **Calibração** — `dl_calibration_tolerance`: zona neutra ON (`neutral_half_width: 0.04`); override TCN macro em raw extremos.
8. **Resolução direcional** — `execution_direction_resolver` + `execution_direction_checks` + `meta_payoff_regression`: edge positivo preserva TCN; edge `< -0.15` em squeeze rebaixa `trade_score=0.52` (`[D-SQUEEZE]`); `ensure_direction_margin` expõe margem corrigida.
9. **Gate de qualidade** — dual soft TCN+meta + HARD microestrutura; starvation a partir de **6** skips; stubs sniper não vetam.
10. **Z-Score meta** — `attach_payoff_edge_zscore_metrics` anexa `meta_payoff_edge_zscore` / `edge_zscore` para ranking e gate.
11. **Deploy** — `deploy_ok=false` bloqueia execução; mini-deploy de treino usa `force_local=True` (modelo em memória).
12. **Seleção** — `market_decision_score` multiplicativo (TCN × fator Z-Score); redirect inter-símbolo quando âncora degradada.
13. **Risco** — Kelly em EXPLORE (`fraction: 0.08`, teto 3,5%); Soft Recovery com cover equilibrado do pending em 2–3 ciclos (`amort_cycles` **2/3**, `cover_multiple` **1.10**, teto `max_safe_stake_pct`); stop win por sessão (4,31% composto ou $10 fixo se banca < $100). Stop loss interno desativado.

Bloqueio absoluto para falhas técnicas (`data`, `predict_error`, `training`, `deploy_ok=false`) e reconciliação pendente. Vetoes HARD de microestrutura bloqueiam independentemente do soft. Não há vetos táticos autônomos de quality guard soft, cooldown pós-LOSS, blackout de broker ou stubs sniper.

Perfil em `config/settings.json` (settings atuais):

| Parâmetro | Valor | Função |
|-----------|-------|--------|
| `calibration.method` | auto | auto + piso sharpness; fallback `identity` se cal colapsar |
| `calibration.neutral_half_width` | 0.0 | Zona neutra **off** |
| `confidence_call_threshold` | 0.51 | Threshold CALL |
| `confidence_put_threshold` | 0.49 | Threshold PUT |
| `dynamic_threshold.enabled` | false | Thresholds flutuantes por volatilidade **desligados** |
| `min_val_accuracy` | 0.60 | Piso de acurácia de validação (treino/deploy) |
| `min_validation_accuracy_gate` | — | Sem piso hard nos settings atuais |
| `min_edge_execute` | 0.0 | Edge base (advisory) |
| `label_mode` | `quantum_multi_barrier` | Rotulagem SSOT; `triple_barrier` / `spot_forward` / `ma_trend` via config |
| `label_vol_window_bars` | 15 | Janela de σ para largura de barreira (tunável por símbolo) |
| `label_vol_multiplier` | 1.0 | Multiplicador da barreira de volatilidade |
| `indicator_gating.*` | removido | Vetos de sinal retirados do codigo (escopo 1) |
| `hard_cal_margin_floor` / `quality_gate.*` / `price_zone.*` / `align_rsi_trend` | removido | Sem rejeicao de sinal/qualidade no pipeline |
| `mandatory_trade_each_cycle` | false | Esteira continua TCN→fusao→Kelly (sem mandato de trade a cada ciclo) |
| `force_trade_every_cycle` | false | Sem sintese forcada |
| `require_meta_for_execution` | false | Meta **opcional** para execucao |
| `bb_width_adaptive_squeeze.enabled` | false | Squeeze adaptativo desligado |
| `loss_protection.min_direction_margin` | 0.0 | Piso de margem na protecao contra loss |
| `loss_protection.max_edge_without_margin` | 999.0 | Cap edge sem margem |
| `loss_protection.max_zscore_without_margin` | 999.0 | Cap Z sem margem |
| `risk_management.kelly.max_stake_pct` | 0.035 | Teto Kelly efetivo |
| `risk_management.kelly.max_bankroll_stake_fraction` | 0.035 | Teto de fração de banca alinhado ao Kelly |
| `risk_management.kelly.fraction` | 0.08 | Fração Kelly base (EXPLORE; compressão 40% fora de recovery) |
| `consensus_penalty_enabled` | false | Consensus Entropy Penalty **desligado** |
| `orchestrator.settlement_tolerance_window_seconds` | 600 | Janela de settlement |
| `orchestrator.watchdog_stale_tick_seconds` | 300 | Watchdog de inanição |

Cover de recovery dimensiona `pending/payout` em **2–3** trades (amort **2/3**, `cover_multiple` **1.10**) para amortizar o passivo, sujeito a `max_safe_stake_*`. Turbo de stake (Z≥1.5) **nunca** ultrapassa `max_safe_stake_cap` pós-multiplicador. Z-Score de edge é bufferizado **por símbolo**. Boot emite `CFG_RISK` via `validate_engine_risk_config` / `RiskPolicy`. Labels train/deploy compartilham `LabelSpec` (`horizon` + `smooth_bars`); treino meta usa proxy de retorno **passado**, não forward.

#### Defaults `risk_management.soft_recovery` (`soft_recovery_policy`)

| Parâmetro | Padrão | Função |
|-----------|--------|--------|
| `enabled` | true | Ativa soft recovery paramétrico |
| `max_safe_stake_cap` | 4.20 | Teto absoluto só em micro-banca a partir de \(N\ge 4\) |
| `max_safe_stake_pct` | 0.05 | Teto percentual de banca em conta ≥ $100 (mandato soft cover) |
| `max_safe_stake_pct_linear2` / `linear3` | 0.045 / 0.04 | Dampen do teto sob stress linear |
| `pending_waives_scale_explore` | true | Pending material libera soft cover sob discord/adapt |
| `amort_cycles_min` / `amort_cycles_max` | 1 / 1 | Cover 100% `pending/payout` em 1 WIN (teto safe) |
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

Com `mandatory_trade_each_cycle: false`, `try_inter_symbol_zscore_redirect` permanece disponivel para redirecionar ranking sem forçar entrada degradada:

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

Em baixo consenso (`retention_raw ≤ consensus_min_retention`, padrão 0,50), a stake é forçada ao piso mínimo da API ($1,00), protegendo contra discordancia entre DL e indicadores clássicos.

**Settings atuais:** `consensus_penalty_enabled: false` — a penalidade permanece no código, mas **não** altera o Kelly base em produção com a config canônica.

**Modo contínuo (quando habilitado):** essa penalidade opera sobre o Kelly base mesmo em recovery. A convergência adaptativa (seção 8.2) evita que a penalidade asfixie a recuperação financeira.

> **Gates defensivos legados removidos / stubs.** O Gate Assimétrico de Proteção, o Micro Noise Gate, o Filtro de Exaustão de Barreira Micro e o Veto de Inversão por Convicção DL foram eliminados. Sniper/Hurst/BB retornam `False`. Em modo mandatário, o quality guard soft permanece telemetria quando a falha é fraca; falha **fortemente negativa** do meta (`predicted_payoff_edge < 0` ou Z `< -0.20` em todos os candidatos) pode suspender o cluster, salvo waiver de emergencia. Vetoes HARD de microestrutura (`adx_starvation`, `vol_ratio_starvation`, `val_accuracy_gate`) bloqueiam independentemente.

---

## 8. Recovery, sizing e persistência financeira

Esta seção documenta as diretrizes matemáticas que corrigem a **inanição por sizing desalinhado**: WINs operacionais com micro-stakes que zeravam o contador de perdas sem extinguir o drawdown real da sessão.

### 8.1 Filosofia de Recovery Financeiro Persistente

#### Problema: reset cego vs. realidade financeira

Em execução continua (`mandatory_trade_each_cycle: false`), o motor pode registrar um **WIN operacional** (P&L positivo no contrato liquidado) enquanto o **saldo acumulado da sessão** (`total_session_profit`) permanece negativo e o **drawdown pendente** (`pending_loss`) ainda carrega valor a recuperar.

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
| **RECOVER** | `pending_total > 0` ou `linear >= 1` | Soft Recovery cover equilibrado (`amort_cycles` **2/3**, teto `max_safe_stake_pct` **3.5%**) | `RECOVER_DAL_Ln` / `D'ALEMBERT` |

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
| `max_safe_stake_pct` | Teto defensivo % banca (default **5%**) |
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
| `amort_cycles_min` / `amort_cycles_max` | 1 / 1 | Cover 100% `pending/payout` em 1 WIN (teto safe) |
| `coing_redirect_drawdown_threshold` | 15.00 | Limiar absoluto (USD) do Consensus Cointegration Redirect |
| Micro-residual | Z floor −0.60; waiver GBDT 6 skips | Baixa intensidade: fecha válvula de cointegração e relaxa veto Z |

O bloco legado `risk_management.dlambert` foi removido da configuração canônica; o motor ainda aceita `dlambert_enabled` apenas como fallback de compatibilidade. A política canônica vive em `soft_recovery_policy.py`.

#### Seleção de símbolo e Hurst em recovery

Complementar à curva soft D'Alembert:

- Rotação de símbolo após loss linear (`symbol_loss_rotation_cycles`); sem bônus fixo em `R_10`
- Filtro de loss-protection com `min_direction_margin: 0.0` (caps edge/Z 999)
- Trava Hurst N2+ (`recovery_hurst_gate`) — prioriza candidatos persistentes; N3+ sem Hurst bloqueia escalada
- Teto de stake comprimido em linear ≥2 (`max_safe_stake_pct_linear2`) e ≥3 (`max_safe_stake_pct_linear3`)

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

Politica de dominio: `domain/analytics/side_equilibrium.py` + store. Runtime live: `execution_side_eq_sizing.apply_side_eq_kelly_sizing` no finalize do direction resolver. Config: `orchestrator.execution.side_equilibrium` (`enabled: true`). O modulo legado `side_equilibrium_gate.py` (hard-skip / flip) **nao** e o path ativo de execucao.

| Regime | Janela | `n_min` | Ação tipica no runtime |
|--------|--------|---------|------------------------|
| **Small-N** | small_window (24) | **8** | Dominio pode emitir `hard_skip`; sizing mapeia para soft `kelly_mult` (sem SKIP de direcao) |
| **Large-N** | 100 trades | 40 | `soft_penalty`: `kelly_mult_soft` (0.55) em `kelly_fraction_scale` |
| Amostra insuficiente | `n_side &lt; n_min` ou `total &lt; n_min` | — | `pass` (log `SIDE_EQ … action=pass`) |

Telemetria: `SIDE_EQ | SYMBOL SIDE | call=W/N put=W/N | bias=… wr=… | action=…` (dedupe por ciclo/simbolo/lado). Outcomes em `process_contract_outcome`. `side_eq_blocked` permanece false; risk/collect nao zeram stake por hard_skip de dominio.

---

## 9. Execução

| Flag | Efeito |
|------|--------|
| `mandatory_trade_each_cycle: false` | Esteira continua TCN→fusao EV→Kelly; signal_skip 1.1 soft; quality gate amplo fora |
| `require_meta_for_execution: false` | Meta opcional |
| `include_anchor_trades` | Inclui âncora nas ordens do cluster |
| `diversify_after_loss_margin` | Prefere símbolo alternativo quando scores são próximos |

Logs: `ord=` (ordem enviada) sempre igual a `dl=` (direção prevista pelo DL), pois não há mais inversão.

---

## 10. Risco e stop win por sessão

| Mecanismo | Papel |
|-----------|-------|
| Kelly fracionário | Sizing EXPLORE com win rate dinâmico; compressão 40% fora de recovery (`fraction: 0.08`) |
| Target Proximity Damping | Amortecimento linear da stake Kelly conforme `pnl_sessao` se aproxima de `target_win` (piso `target_damping_floor` **0.50** + span **0.50**; arranque **1.0**) |
| Consensus Entropy Penalty | Presente no código; **desligado** nos settings atuais (seção 7) |
| Penalty Smoothing | Convergência adaptativa em recovery quando consensus estiver ligado (seção 8.2) |
| Recovery financeiro persistente | Estado de risco atrelado a `pending_total` (seção 8.1) |
| Sizing | Kelly EXPLORE + Soft Recovery RECOVER (seção 8.3) |
| Side equilibrium (LLN) | Small-N / large-N CALL/PUT (seção 8.4) |
| Stop win por sessão ativa | Banca ≥ $100: `target_win = session_start_balance × compounding_rate_daily` (padrão 3,00%); banca < $100: stop win fixo `$10`; fast-path anti-deadlock |
| Stop loss | Desativado — sem reset por relógio nem disjuntor de perda diária |

### 10.1 Juros compostos e controle operacional

A meta segue a planilha de gerenciamento de juros compostos (`compounding_rate_daily: 0.03`), com override de micro-banca:

| Evento | Comportamento |
|--------|---------------|
| Boot do processo | Captura saldo Deriv (ou `session_start_balance` em settings) como `session_start_balance` |
| Meta calculada (banca ≥ $100) | `target_win = session_start_balance × 0,03` (arredondada para baixo em centavos) |
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
| `compounding_rate_daily` | `0.0431` | Taxa de juros (4,31% sobre banca inicial — Single-Strike) |
| `session_start_balance` | `null` | Override manual da banca inicial (senão usa saldo Deriv) |
| `small_account_threshold` | `100.0` | Limiar abaixo do qual o stop win é fixo |
| `small_account_stop_win` | `10.0` | Stop win fixo em dólares para micro-banca |
| `duration` | `5` | Duração do contrato RISE_FALL (**m**); ops fixo via `ops_contract_duration_minutes`; ciclo **120 s** / micro OHLC **300 s** (M5); label TCN = `label_horizon_bars` (**1** vela M5; grade sweep H1–H4 em M5) |

Com `compounding_enabled: false`, o motor recorre ao alvo legado (`small_account_stop_win` / `large_account_stop_win_pct`).

### 10.2 Sizing defensivo de proximidade de alvo

Evita superexposição quando a sessão já capturou a maior parte do stop win:

1. **Kelly base** — `resolve_effective_kelly_fraction` usa `kelly.fraction` de config (**0,08**), com target proximity em regime EXPLORE (consensus off).
2. **Amortecimento dinâmico** — após o Kelly bruto, `apply_kelly_target_proximity_damping` multiplica a stake por `target_damping_floor + target_damping_span × remaining_target_pct` (**0.50 + 0.50 × remaining**; arranque **1.0**, perto-meta **0.50**).
3. **Exemplo** — meta $101.20, Kelly bruto $45.56: com `pnl_sessao = 0` permanece $45.56×1.0 (já atenuado pela fração base); com 90% da meta (`pnl ≈ $91.08`) o fator cai para 0.46 (~$20.96).

Fora de recovery, este amortecimento define o `Kelly_base`. Em RECOVER com Soft Recovery, a stake amortiza `pending` sob `max_safe_stake_pct`.

Log de bootstrap (banca ≥ $100): `SESSAO INICIADA | Alvo de 3,00%: $XX.XX | Stop Loss: DESATIVADO`.

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
Com universo single-symbol (`R_10`), o lock contra drift natural de par Bull/Bear permanece no codigo como no-op (`hedge_peer(R_10)` retorna `None`). Quando houver par de hedge configurado, permanece vedada a emissao contra o drift sob expansao hiperbolica de volatilidade (\(Z_{\text{vol}} \ge 2.0\)):
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
- Sob **Micro Passivo Residual** (banca ≤ $250, `pending_total` ≤ $5 e < 5% da banca), o piso do veto `meta_payoff_negative_zscore_veto` relaxa de \(Z \ge -0.20\) para \(Z \ge -0.60\), a válvula de cointegração permanece fechada e o waiver GBDT antecipa para 6 skips — evitando loops `EXEC_EMPTY` em passivo centavado. Após `EXEC_EMPTY` em recovery, o orquestrador alinha o cooldown à fronteira de assinatura de **300 s** (M5; sem retries fragmentados de 8s).

---

## 11. Referências internas

- [arquitetura.md](arquitetura.md)
- [structure.md](structure.md) — inventário completo dos módulos Python em `app/src/` (~226)
- [README.md](../README.md)
- [CHANGELOG.md](CHANGELOG.md)
