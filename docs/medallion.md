# Metodologia quantitativa

O Aether Quantum Engine herda a postura **Medallion** no sentido operacional: o mercado é um **sistema de sinais ruidosos**, não uma narrativa macro discricionária. A implementação concentra-se nos índices **Drift** (`RDBEAR`, `RDBULL`) com **Deep Learning** e classificação binária Rise/Fall.

Para arquitetura de código, ver [`arquitetura.md`](arquitetura.md).

---

## 1. Princípios

| Princípio | No motor atual |
|-----------|----------------|
| Sinais, não histórias | Direção CALL/PUT estritamente pela TCN (`P(CALL) > P(PUT)`) |
| Horizonte curto | Contexto DL **M15 (900 s)**; execução **M1 (60 s)**; label `triple_barrier` (ou `ma_trend` legado) |
| Esteira mandatária | `mandatory_trade_each_cycle: true` — mandatory pick quando pool DL aprovado pelo quality gate |
| Modelo pronto antes de operar | `FASE TREINO` suspende ordens até treino da sessão |
| Operação configurável | `mandatory_trade_each_cycle` — esteira mandatária contínua (padrão atual) |
| Feedback real | Win rate live misturado em `val_accuracy`; retreino após loss |
| Defesa contra ruído CSPRNG | Consensus Entropy Penalty no Kelly base |
| Persistência financeira | Recovery atrelado a `pending_loss`, não a WIN operacional isolado |
| Martingale Geométrico sem teto | Em recovery, `Stake = Kelly_base × 2^n` escalando até recuperar o passivo total |
| Meta por sessão ativa | Stop win de 2,60% composto sobre banca inicial; operador controla quantas sessões por dia |
| Sem disjuntor de perda | Stop loss interno desativado; recovery geométrico sem teto de nível, stake ou drawdown |
| Isolamento de estado | `asyncio.Lock` serializa inferência, liquidação e persistência; elimina race conditions pós-reset linear |

---

## 2. Universo Drift e perfil de qualidade

### 2.1 Universo Drift

Índices sintéticos correlacionados no eixo de barreiras. Cada símbolo tem modelo DL independente com **34 features** e volatilidade calibrada ao alvo do índice.

| Símbolo | Papel típico |
|---------|----------------|
| `RDBULL` | Âncora padrão; referência de cluster; bônus de ranking em recovery |
| `RDBEAR` | Par correlacionado; candidato a redirect inter-símbolo quando âncora degradada |

Operação: contratos **RISE_FALL** (CALL = alta no período, PUT = queda).

### 2.2 Telemetria de Volatilidade, Exaustão e Fluxo Micro

Indicadores micro de 60 s (RSI, `vol_ratio`, Keltner, `bb_width`, aceleração de ticks) alimentam o container `aether-meta-classifier` via vetor **39D**. O `LGBMRegressor` (huber) estima `predicted_payoff_edge` contínuo; o resolver preserva score orgânico da TCN quando o edge é positivo e aciona downgrade D-SQUEEZE quando o edge colapsa em compressão M1.

**Spread de convicção cross-symbol** (triplet anexado em `prepare_meta_classifier_cross_symbol_bundle`):

| Feature | Descrição |
|---------|-----------|
| `cross_symbol_prob_delta` | `abs(P(CALL)_RDBULL − P(PUT)_RDBEAR)` — divergência de convicção entre índices |
| `cross_symbol_vol_ratio_diff` | Spread linear M1 de `vol_ratio` (BULL − BEAR) |
| `cross_symbol_rsi_spread` | Spread linear M1 de RSI micro (BULL − BEAR) |

Em regimes de drift paralelo (ambos símbolos com scores altos na mesma direção), spreads baixos sinalizam saturação espelhada — o GBDT usa isso para evitar entradas sem viés relativo.

Features de fluxo extraídas do `TickBuffer`:

| Feature | Descrição |
|---------|-----------|
| `micro_tick_acceleration` | Aceleração estocástica de ticks nos últimos 5 s do minuto corrente |
| `keltner_deviation_ratio` | Distância fracionária do último tick ao centro do canal Keltner micro |

Indicadores macro (Hurst, ADX, bandas) permanecem em `metrics["indicators"]` / `feature_vector` (34D TCN) como telemetria analítica e insumo do stacking — sem veto direcional autônomo fora do meta-classificador.

### 2.5 Perfil de qualidade atualizado

| Camada | Comportamento |
|--------|---------------|
| Bloqueio técnico | `data`, `predict_error`, `training`, `deploy_ok=false` |
| Classificação macro | TCN processa lookback de 12 h em M15; define direção (`dl_direction`) |
| Stacking tabular | Meta-regressor LightGBM (M1) sobre vetor **39D** + probabilidade TCN; saída `predicted_payoff_edge` |
| Z-Score de payoff | `payoff_edge_zscore`: janela adaptativa 15–45; classificação estatística do micro-edge |
| Scoring de ranking | `market_decision_score = tcn × max(0.1, 1 + z)` — penaliza sinais `NO_EDGE_NEUTRAL` / `LOSS_EXPECTED` |
| Scoring direcional | TCN define `dl_direction`; edge `> 0` mantém score orgânico; edge `< -0.15` em squeeze rebaixa para `0.52` |
| Margem direcional | `direction_margin = abs(P(lado_escolhido) − 0.50)`; CALL usa `calibrated_prob`; PUT usa `1 − prob` |
| Gate de qualidade | Dual TCN (`passes_execution_quality`) + meta Z-Score (`evaluate_meta_payoff_quality`); suspensão cooperativa via `quality_conviction_suspends_cluster` |
| Rotulagem Triple Barrier | Barreira dinâmica por σ de ticks; neutro em lateralização; tunável por símbolo |
| Perda TCN assimétrica | Penalidade 2,5× para erro direcional em alta volatilidade |
| Optuna meta | Maximiza Information Ratio; constraint OOS payoff Z-Score ≥ +1,00 |
| Gerenciamento de risco | Kelly base + Martingale Geométrico puro (`Kelly_base × 2^n`) sem teto macro |

---

## 3. Blindagem multi-timeframe

| Camada | Timeframe | Papel |
|--------|-----------|-------|
| Deep Learning / TCN | M15 (900 s) | Tensor `[1, 48, 34]` = 12 h de contexto macro |
| Meta-regressor GBDT | M1 (60 s) | Regressão tabular **39D**; edge contínuo + downgrade D-SQUEEZE |
| Orquestrador / contrato | M1 (60 s) | Ciclo a cada minuto; RISE_FALL de 60 s |
| Resolução direcional | TCN + meta GBDT | `dl_direction` da TCN; `exec_direction` permanece alinhada salvo bloqueio técnico |
| Execução contínua | M1 | Boleta CALL/PUT na virada do minuto sempre que há sinal válido |

Com `granularity: 900` (M15) e `training_history_bars: 15552`:

| Conceito | Barras | Tempo aproximado |
|----------|--------|------------------|
| Histórico de treino | 15552 | ~162 dias |
| Lookback | 48 | **12 h** de contexto por sequência |
| Validação holdout | ~15% | proporcional ao split |

---

## 4. Camadas de decisão (qualidade)

Ordem lógica de uma entrada:

1. **Fase** — todos os modelos com treino da sessão concluído.
2. **Dados** — histórico suficiente (`gate_reason=data`).
3. **Treinamento** — modelo do símbolo treinado na sessão (`gate_reason=training`).
4. **Predição DL** — inferência Triton concorrente; `raw_prob`/`calibrated_prob` e indicadores calculados.
5. **Bundle cross-symbol** — `prepare_meta_classifier_cross_symbol_bundle` coleta telemetria micro M1 em paralelo e anexa spreads cross-symbol.
6. **Stacking tabular** — `MetaClassifierClient` envia probabilidade TCN + vetor **39D** ao `aether-meta-classifier`; retorna `predicted_payoff_edge`.
7. **Resolução direcional** — `execution_direction_resolver` + `meta_payoff_regression`: edge positivo preserva TCN; edge `< -0.15` em squeeze rebaixa `trade_score=0.52` (`[D-SQUEEZE]`); `ensure_direction_margin` expõe margem corrigida.
8. **Gate de qualidade** — dual TCN (`passes_execution_quality`) + meta Z-Score (`evaluate_meta_payoff_quality`); suspensão cooperativa do cluster
9. **Z-Score meta** — `attach_payoff_edge_zscore_metrics` classifica expectativa (`WIN_EXPECTED` se Z ≥ 0,50 e edge > 0).
10. **Deploy** — `deploy_ok=false` bloqueia execução.
11. **Seleção** — `market_decision_score` multiplicativo (TCN × fator Z-Score); redirect inter-símbolo quando âncora degradada.
12. **Risco** — Kelly + Consensus Penalty; recovery financeiro persistente; Martingale Geométrico `Kelly_base × 2^n`; stop win por sessão ativa (2,60% composto).

Bloqueio absoluto para falhas técnicas (`data`, `predict_error`, `training`, `deploy_ok=false`) e reconciliação pendente. Não há vetos táticos autônomos de quality guard, cooldown pós-LOSS, blackout de broker ou Hurst em recovery.

Perfil em `config/settings.json`:

| Parâmetro | Valor | Função |
|-----------|-------|--------|
| `calibration.method` | auto | Platt, isotonic ou temperatura+Platt no holdout |
| `confidence_call_threshold` | 0.53 | Base de calibração CALL |
| `confidence_put_threshold` | 0.47 | Base de calibração PUT |
| `dynamic_threshold.enabled` | true | Thresholds flutuantes por volatilidade |
| `min_val_accuracy` | 0.60 | Piso de acurácia de validação |
| `min_edge_execute` | 0.04 | Edge base (advisory) |
| `label_mode` | `triple_barrier` | Rotulagem por barreira tripla (recomendado para M1) |
| `label_vol_window_bars` | 15 | Janela de σ para largura de barreira (tunável por símbolo) |
| `label_vol_multiplier` | 1.0 | Multiplicador da barreira de volatilidade |
| `quality_gate.regular.min_direction_margin` | 0.06 | Piso legado (observabilidade; não veta em modo mandatário) |
| `quality_gate.regular.min_payoff_edge` | 0.01 | Piso legado (observabilidade) |
| `quality_gate.min_direction_margin` | 0.12 | Piso legado recovery (observabilidade) |
| `quality_gate.min_payoff_edge` | 0.04 | Piso legado recovery (observabilidade) |
| `mandatory_trade_each_cycle` | true | Esteira mandatária contínua |
| `consensus_penalty_enabled` | true | Atenua Kelly quando ord diverge dos votos |
| `penalty_smoothing_factor` | 0.40 | Suavização convexa em recovery com trade_score > 0.68 |

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

Classificação do buffer (`payoff_edge_zscore.py`):

| Classe | Condição |
|--------|----------|
| `WIN_EXPECTED` | `edge > 0` e `z ≥ 0,50` |
| `NO_EDGE_NEUTRAL` | `edge > 0` e `z < 0,50` |
| `LOSS_EXPECTED` | `edge ≤ 0` |

O buffer histórico **não** avança durante recovery ou com `linear_losses > 0`, evitando contaminação estatística em Martingale.

### 5.1 Redirect inter-símbolo (modo contínuo)

Com `mandatory_trade_each_cycle: true`, `try_inter_symbol_zscore_redirect` evita inanição operacional sem forçar entrada degradada:

| Gatilho | Limiar |
|---------|--------|
| Âncora degradada | `meta_payoff_edge_zscore < -0.50` |
| Par alternativo forte | `meta_payoff_edge_zscore > +0.50` |

O fluxo inteiro do ciclo desvia para o par com maior probabilidade matemática de ganho no milissegundo de entrada.

---

## 6. Resolução direcional com edge contínuo e downgrade D-SQUEEZE

`execution_direction_resolver.resolve_execution_direction` delega o refinamento a `meta_payoff_regression.apply_meta_regression_edge`:

| Etapa | Regra |
|-------|-------|
| `dl_direction` | TCN: `P(CALL) > pivot` → CALL, caso contrário PUT |
| Payoff GBDT | `predicted_payoff_edge` do container LightGBM (porta 8005) |
| Edge positivo | `predicted_payoff_edge > 0.0` → `exec_direction = dl_direction`; `trade_score` orgânico da TCN |
| Downgrade squeeze | `predicted_payoff_edge < -0.15` **e** (`bb_width < 0.06` **ou** `micro_tick_acceleration < 0`) → `trade_score=0.52`; `meta_squeeze_downgrade=true`; log `[D-SQUEEZE]` |
| Edge negativo leve | Mantém `dl_direction` e score orgânico |
| `direction_margin` | `abs(P(lado_escolhido) − 0.50)` — distância ao neutro; recalculada por `ensure_direction_margin` no retorno do resolver |
| `direction_inverted` | `False` no fluxo de regressão |

**Justificativa do score 0.52 em squeeze**: em canais Bollinger esmagados (ex.: ciclos C0014–C0016 com `bb_width` 0.03–0.09), o rebaixamento força o `consensus_stake_penalty` a comprimir a stake ao piso de $1.00 da Deriv, curto-circuitando a cauda exponencial do Martingale Geométrico nos frames de maior ruído estocástico.

`execution_direction_cross_corr` e `execution_volatility_booster` permanecem como telemetria consultiva.

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

**Modo contínuo:** essa penalidade opera sobre o Kelly base mesmo quando o motor já está em recovery Martingale Geométrico. A convergência adaptativa (seção 8.2) evita que a penalidade asfixie a recuperação financeira.

> **Gates defensivos legados removidos.** O Gate Assimétrico de Proteção, o Micro Noise Gate, o Filtro de Exaustão de Barreira Micro e o Veto de Inversão por Convicção DL foram eliminados. Em modo mandatário, portões secundários (quality guard, cooldown pós-LOSS, blackout de broker, Hurst em recovery) retornam sempre não bloqueantes — a esteira só para em falha técnica, reconciliação pendente, stop win ou settlement ativo.

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
| `consecutive_losses_linear` | Contador de clusters negativos — **memória operacional** de stress; expoente do Martingale Geométrico |
| `dlambert_unit` (U) | Unidade aditiva capturada na primeira stake Kelly da sessão (ou override de config) |
| `recovery_financially_active` | Verdadeiro iff `pending_total > 0` |

#### Regra de persistência (implementação atual)

O motor **não utiliza reset cego** de `consecutive_losses_linear` baseado em WIN operacional isolado.

| Condição após liquidação | Comportamento |
|--------------------------|---------------|
| `cluster_profit < 0` | `consecutive_losses_linear += 1`; `pending_loss` incrementado |
| `cluster_profit ≥ 0` **e** `pending_total > 0` | WIN absorvido no drawdown; **`consecutive_losses_linear = max(1, n-1)`** (retração do Martingale) |
| `cluster_profit ≥ 0` **e** `pending_total = 0` | Recovery financeiro extinto; reset de `consecutive_losses_linear` e `last_loss_stake` |

**Persistência de Drawdown:** o robô permanece em estado de Recovery (Martingale Geométrico `Kelly_base × 2^n`) até que `pending_total` seja **financeiramente zerado** por retornos reais da API — não por um WIN simbólico que não cobre o buraco acumulado.

#### Implicação para gestão de cauda

- O **estado de risco** segue o **passivo financeiro** (`pending_loss`), não a contagem superficial de vitórias.
- Micro-WINs em recovery **amortizam** o drawdown e **retraem** o expoente da curva (`max(1, n-1)`), mas **não encerram** o regime de recovery prematuramente.
- Logs de auditoria: `RISK: WIN operacional`, `RISK: Lucro parcial`, `RISK: Recovery financeiro zerado` — cada um com `pend=$` e `pnl_sess=$`.

---

### 8.2 Waiver de Consensus Penalty em Recovery

#### Problema: penalidade convexa vs. peso financeiro do recovery

O **Consensus Entropy Penalty** comprime `f*` quando a ordem diverge dos votos técnicos. Em recovery, punir a stake por falta de consenso upstream asfixia a recuperação — o Martingale Geométrico precisa de peso financeiro real para extinguir o passivo.

#### Bypass de consensus em recovery com passivo pendente

Quando `pending_total > 0`, `risk_stake_calc.py` **ignora** a penalidade de entropia de votação e o piso `stake_min` por divergência de consenso. O payload segue direto para Martingale Geométrico com tag `D'ALEMBERT`.

#### Condições de waiver absoluto (`retention = 1.0`)

1. **Recovery ativo:** `pending_total > 0` **ou** `consecutive_losses_linear > 0`
2. **Qualquer** candidato do cluster em recovery que atenda **uma** das condições:
   - **Votos unânimes alinhados:** `6×0` ou `0×6` na direção da ordem (M15)
   - **Convicção elevada:** `trade_score >= penalty_smoothing_trade_score_min` (padrão **0,68**)

Justificativa: com alinhamento direcional unânime em M15 ou convicção alta, o Kelly base não pode ser esmagado pela penalidade de entropia — a curva geométrica precisa operar com peso financeiro real em símbolos secundários do cluster (ex.: `RDBEAR`).

---

### 8.3 Martingale Geométrico Puro sem teto (Kelly base × 2^n)

#### Filosofia: recuperação exponencial do passivo

Em recovery ativo, o sizing abandona qualquer escada aditiva e adota a **curva multiplicativa clássica**: cada LOSS dobra a exposição, escalando geometricamente até que um WIN cubra o passivo acumulado. Não há teto de nível linear, múltiplo de stake nem drawdown de sessão.

#### Fórmula de stake (`dlambert_sizing.geometric_martingale_stake`)

| Estado | Stake |
|--------|-------|
| Normal (`recovery_active` falso) | Kelly fracionário (+ booster super-concordance se P≥0.75, 6×0, Hurst>0.55), tag `KELLY` |
| Recovery (`pending_total > 0` ou `consecutive_losses_linear > 0`) | `Effective_Base × 2^consecutive_losses_linear`, tag `D'ALEMBERT` |

```
Effective_Base = max(dlambert_unit_override, U)
GEOMETRIC_MARTINGALE_BASE = 2.0
stake_raw = Effective_Base × 2^max(0, consecutive_losses_linear)
```

| Termo | Significado |
|-------|-------------|
| `U` (`dlambert_unit`) | Primeira stake Kelly capturada na sessão |
| `Effective_Base` | Piso ancorado: nunca usa Kelly comprimido por consenso em recovery |
| `consecutive_losses_linear` | Contador de stress; expoente da curva |

#### Retração em WIN parcial

```
WIN parcial (pending_total > 0 após liquidação):
  consecutive_losses_linear = max(1, n - 1)   → reduz o expoente da curva

WIN total (pending_total = 0):
  consecutive_losses_linear = 0               → volta ao Kelly puro
```

O expoente **sobe** uma unidade por LOSS e **desce** uma por WIN parcial, nunca abaixo de 1 enquanto recovery ativo — evitando reset cego sem extinguir o drawdown.

#### Sem circuit breaker

Foram removidos `dlambert_circuit_breaker`, as constantes `MAX_LINEAR_LEVEL`, `MAX_STAKE_U_MULTIPLE`, `MAX_SESSION_DRAWDOWN_U` e o curto-circuito da tag `D'ALEMBERT_CB` em `calculate_stake_for_manager`. O cálculo prossegue livre na thread principal, sem forçar stake a `0.0` nem travar exposição.

#### Exemplo numérico

```
Sessão: Kelly_base = $8

Kelly puro:     n=0 → stake = $8 × 2^0 = $8
LOSS #1:        n=1 → stake = $8 × 2^1 = $16
LOSS #2:        n=2 → stake = $8 × 2^2 = $32
LOSS #3:        n=3 → stake = $8 × 2^3 = $64
WIN parcial:    n=2 → stake = $8 × 2^2 = $32
WIN total:      n=0 → stake = $8 (Kelly puro)
```

#### Parâmetros de configuração (`risk_management.dlambert`)

| Parâmetro | Padrão | Função |
|-----------|--------|--------|
| `dlambert_enabled` | true | Ativa Martingale Geométrico em recovery |
| `dlambert_unit_override` | null | Força base fixa (ignora captura Kelly) |

#### Seleção de símbolo e Hurst em recovery

Complementar à curva geométrica:

- Ranking com diversificação e bônus em `RDBULL`/`RDBEAR`
- Trava Hurst N2+ (`recovery_hurst_gate`) — piso logarítmico de score

---

### 8.4 Barreira de persistência pós-reset linear

Quando um cluster encerra com reset linear D'Alembert (`_linear_reset_occurred`), o motor executa uma sequência atômica antes de permitir novo ciclo de inferência:

1. `_finalize_linear_reset_risk_state` — zera `consecutive_losses_linear`, `last_loss_stake` e limpa entradas zeradas de `pending_loss`.
2. `_persist_session_state_snapshot` — espelha saldo em `StateManager` e persiste `data/session_state.json`.
3. `_persist_full_state_unlocked` — bundle Redis sem reentrância no lock.
4. Yield cooperativo de **0,1 s** — libera o event loop antes do próximo ciclo DL.

Durante a barreira, `session_persistence_write_active` impede que `trading_cycle_entry` execute `execute_cluster` concorrentemente. O lock central (`StateManager._state_lock`) garante que leituras de stake e escrita de liquidação não se sobreponham — eliminando deadlocks térmicos silenciosos observados na transição entre clusters (ex.: C0006 → inferência subsequente).

---

## 9. Execução

| Flag | Efeito |
|------|--------|
| `mandatory_trade_each_cycle: true` | Esteira contínua — toda oportunidade DL tecnicamente válida segue para execução; redirect inter-símbolo se âncora Z<-0.50 e par Z>+0.50 |
| `mandatory_trade_each_cycle: false` | Modo legado seletivo (não recomendado na configuração atual) |
| `include_anchor_trades` | Inclui âncora nas ordens do cluster |
| `diversify_after_loss_margin` | Prefere símbolo alternativo quando scores são próximos |

Logs: `ord=` (ordem enviada) sempre igual a `dl=` (direção prevista pelo DL), pois não há mais inversão.

---

## 10. Risco e stop win por sessão

| Mecanismo | Papel |
|-----------|-------|
| Kelly fracionário | Sizing base com win rate dinâmico após amostras mínimas; compressão estática de 60% fora de recovery |
| Target Proximity Damping | Amortecimento linear da stake Kelly conforme `pnl_sessao` se aproxima de `target_win` (piso 0.40×) |
| Consensus Entropy Penalty | Defesa contra ruído CSPRNG (seção 6) |
| Penalty Smoothing | Convergência adaptativa em recovery (seção 8.2) |
| Recovery financeiro persistente | Estado de risco atrelado a `pending_total` (seção 8.1) |
| Martingale Geométrico sem teto | Recuperação exponencial `Kelly_base × 2^n` (seção 8.3) |
| Stop win por sessão ativa | `target_win = session_start_balance × compounding_rate_daily` (padrão 2,60%); fast-path anti-deadlock |
| Stop loss | Desativado — sem reset por relógio nem disjuntor de perda diária |

### 10.1 Juros compostos e controle operacional

A meta segue a planilha de gerenciamento de juros compostos (`compounding_rate_daily: 0.026`):

| Evento | Comportamento |
|--------|---------------|
| Boot do processo | Captura saldo Deriv (ou `session_start_balance` em settings) como `session_start_balance` |
| Meta calculada | `target_win = session_start_balance × 0,026` (arredondada para baixo em centavos) |
| Durante a sessão | `pnl_sessao = current_balance - session_start_balance` |
| Meta atingida | Fast-path: `clear_current_session_redis_keys` → `cancel_settlement_queue_fast` → `graceful_shutdown(fast_path=True)` |
| Ciclo pós-liquidação incompleto | Retry com breath; após 2 falhas consecutivas → `emergency_save_session_state` + `recover_post_settlement_loop_transparently` |
| Restart manual | Nova sessão independente com novo saldo e nova meta de 2,60% |
| Mesmo dia civil | Múltiplas sessões isoladas permitidas — sem virada UTC/meia-noite |

Parâmetros em `risk_management.params`:

| Chave | Padrão | Função |
|-------|--------|--------|
| `compounding_enabled` | `true` | Ativa meta composta por sessão |
| `compounding_rate_daily` | `0.026` | Taxa de juros (2,60% sobre banca inicial) |
| `session_start_balance` | `null` | Override manual da banca inicial (senão usa saldo Deriv) |

Com `compounding_enabled: false`, o motor recorre ao alvo legado (`small_account_stop_win` / `large_account_stop_win_pct`).

### 10.2 Sizing defensivo de proximidade de alvo

Evita superexposição quando a sessão já capturou a maior parte do stop win de 2,60%:

1. **Kelly base comprimido** — `resolve_effective_kelly_fraction` aplica retenção de 40% (`fraction` de config `0.0035` → coeficiente `0.0012`), ancorando `Kelly_base` na faixa ~$10–$12 em vez de ~$31.
2. **Amortecimento dinâmico** — após o Kelly bruto, `apply_kelly_target_proximity_damping` multiplica a stake por `0.40 + 0.60 × remaining_target_pct`.
3. **Exemplo** — meta $101.20, Kelly bruto $45.56: com `pnl_sessao = 0` permanece $45.56×1.0 (já atenuado pela fração base); com 90% da meta (`pnl ≈ $91.08`) o fator cai para 0.46 (~$20.96).

Fora de recovery, este amortecimento define o `Kelly_base`. Em recovery, o Martingale Geométrico `Kelly_base × 2^n` opera sem amortecimento de proximidade.

Log de bootstrap: `SESSAO INICIADA | Alvo de 2,60%: $XX.XX | Stop Loss: DESATIVADO`.

---

## 11. Referências internas

- [arquitetura.md](arquitetura.md)
- [structure.md](structure.md) — inventário completo dos 208 módulos Python
- [README.md](../README.md)
- [CHANGELOG.md](CHANGELOG.md)
