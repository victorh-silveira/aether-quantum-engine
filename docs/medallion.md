# Metodologia quantitativa

O Aether Quantum Engine herda a postura **Medallion** no sentido operacional: o mercado é um **sistema de sinais ruidosos**, não uma narrativa macro discricionária. A implementação concentra-se nos índices **Drift** (`RDBEAR`, `RDBULL`) com **Deep Learning** e classificação binária Rise/Fall.

Para arquitetura de código, ver [`arquitetura.md`](arquitetura.md).

---

## 1. Princípios

| Princípio | No motor atual |
|-----------|----------------|
| Sinais, não histórias | Direção CALL/PUT estritamente pela TCN (`P(CALL) > P(PUT)`) |
| Horizonte curto | Contexto DL **M15 (900 s)**; execução **M1 (60 s)**; label `ma_trend` |
| Boletamento contínuo | Gate de qualidade neutro: sinal válido opera sempre, sem veto nem skip de ciclo |
| Modelo pronto antes de operar | `FASE TREINO` suspende ordens até treino da sessão |
| Operação configurável | `mandatory_trade_each_cycle`: seletivo (`false`) ou contínuo (`true`) |
| Feedback real | Win rate live misturado em `val_accuracy`; retreino após loss |
| Defesa contra ruído CSPRNG | Consensus Entropy Penalty no Kelly base |
| Persistência financeira | Recovery atrelado a `pending_loss`, não a WIN operacional isolado |
| Martingale Geométrico sem teto | Em recovery, `Stake = Kelly_base × 2^n` escalando até recuperar o passivo total |
| Meta por sessão ativa | Stop win de 1% composto sobre banca inicial; operador controla quantas sessões por dia |
| Sem disjuntor de perda | Stop loss interno desativado; recovery geométrico sem teto de nível, stake ou drawdown |

---

## 2. Universo Drift e perfil de qualidade

### 2.1 Universo Drift

Índices sintéticos correlacionados no eixo de barreiras. Cada símbolo tem modelo DL independente com **34 features** e volatilidade calibrada ao alvo do índice.

| Símbolo | Papel típico |
|---------|----------------|
| `RDBULL` | Âncora padrão; referência de cluster |
| `RDBULL` / `RDBEAR` | Núcleo do cluster; bônus em recovery |
| `RDBEAR` / `RDBULL` | Pares de hedge para recovery |

Operação: contratos **RISE_FALL** (CALL = alta no período, PUT = queda).

### 2.2 Telemetria de Volatilidade, Exaustão e Fluxo Micro

Indicadores micro de 60 s (RSI, `vol_ratio`, Keltner, aceleração de ticks) alimentam o container `aether-meta-classifier` via vetor **39D**. O LightGBM recalibra `calibrated_payoff_score` e, quando detecta saturação severa de topo/fundo (`payoff < 0.42`), o resolver **inverte** `exec_direction` para a contra-tendência micro (`meta_direction_flip=true`, `trade_score=0.75`).

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
| Classificação macro | TCN processa lookback de 12 h em M15; define direção estrita (`exec_direction`) |
| Stacking tabular | Meta-classificador LightGBM (M1) sobre vetor **39D** + probabilidade TCN |
| Scoring direcional | TCN define `dl_direction`; meta-classificador pode inverter `exec_direction` em exaustão micro (`payoff < 0.42`) |
| Gate de qualidade | Neutro: participa sempre do pool, sem skip por comportamento |
| Gerenciamento de risco | Kelly base + Martingale Geométrico puro (`Kelly_base × 2^n`) sem teto macro |

---

## 3. Janela temporal de treino

## 3. Blindagem multi-timeframe

| Camada | Timeframe | Papel |
|--------|-----------|-------|
| Deep Learning / TCN | M15 (900 s) | Tensor `[1, 48, 34]` = 12 h de contexto macro |
| Meta-classificador GBDT | M1 (60 s) | Stacking tabular **39D** (34 TCN + 3 cross-symbol + 2 fluxo); inversão micro quando `payoff < 0.42` |
| Orquestrador / contrato | M1 (60 s) | Ciclo a cada minuto; RISE_FALL de 60 s |
| Resolução direcional | TCN + meta GBDT | `dl_direction` da TCN; `exec_direction` pode inverter em saturação micro |
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
5. **Stacking tabular** — `MetaClassifierClient` envia probabilidade TCN + vetor **39D** ao `aether-meta-classifier`; retorna `calibrated_payoff_score`.
6. **Resolução direcional** — `execution_direction_resolver`: inverte `exec_direction` quando `payoff < 0.42` (exaustão Keltner/Bollinger micro); `meta_direction_flip=true`.
7. **Gate de qualidade** — neutro: `passes_execution_quality` retorna `True` e `regime_skip_cycle=False` invariável.
8. **Deploy** — `deploy_ok=false` bloqueia execução.
9. **Seleção** — `market_decision_score` entre candidatos elegíveis.
10. **Risco** — Kelly + Consensus Penalty; recovery financeiro persistente; Martingale Geométrico `Kelly_base × 2^n`; stop win por sessão ativa (1% composto).

Bloqueio absoluto **somente** para falhas técnicas (`data`, `predict_error`, `training`, `deploy_ok=false`). Não há vetos táticos, inversões nem skip por qualidade: qualquer sinal válido participa do pool.

Perfil em `config/settings.json`:

| Parâmetro | Valor | Função |
|-----------|-------|--------|
| `calibration.method` | auto | Platt, isotonic ou temperatura+Platt no holdout |
| `confidence_call_threshold` | 0.53 | Base de calibração CALL |
| `confidence_put_threshold` | 0.47 | Base de calibração PUT |
| `dynamic_threshold.enabled` | true | Thresholds flutuantes por volatilidade |
| `min_val_accuracy` | 0.60 | Piso de acurácia de validação |
| `min_edge_execute` | 0.04 | Edge base (advisory) |
| `mandatory_trade_each_cycle` | false | `true` = modo contínuo (uma ordem/ciclo) |
| `consensus_penalty_enabled` | true | Atenua Kelly quando ord diverge dos votos |
| `penalty_smoothing_factor` | 0.40 | Suavização convexa em recovery com trade_score > 0.68 |

---

## 5. Resolução direcional com inversão micro

`execution_direction_resolver.resolve_execution_direction` aplica a matriz de inversão por probabilidade do meta-classificador:

| Etapa | Regra |
|-------|-------|
| `dl_direction` | TCN: `P(CALL) > pivot` → CALL, caso contrário PUT |
| Payoff GBDT | `calibrated_payoff_score` do container LightGBM (porta 8005) |
| Inversão | `payoff < 0.42` → `exec_direction` oposto; `trade_score=0.75`; `meta_direction_flip=true` |
| Sem inversão | `exec_direction = dl_direction`; `trade_score` recalibrado pelo payoff |
| `direction_inverted` | `True` apenas quando `meta_direction_flip=true` |

`execution_direction_cross_corr` e `execution_volatility_booster` permanecem como telemetria consultiva.

---

## 6. Kelly e Consensus Entropy Penalty (base)

Quando a ordem final (`order_direction`) diverge da maioria dos votos técnicos (`call_votes`/`put_votes`):

| Etapa | Definição |
|-------|-----------|
| Concordância | `agreement = votos_alinhados / (call_votes + put_votes)` |
| Divergência | `divergence = 1 - agreement` |
| Penalidade convexa | `penalty = divergence^exponent × (w_di·|di_opp| + w_cmo·|cmo_opp| + w_rsi·|rsi_opp|)` |
| Retenção Kelly | `retention_raw = max(floor, 1 - min(max_cut, penalty))` |
| Stake Kelly | `f*_efetivo = f* × retention` |

Em baixo consenso (`retention_raw ≤ consensus_min_retention`, padrão 0,50), a stake é forçada ao piso mínimo da API ($1,00), protegendo contra ruído do CSPRNG quando DL e indicadores clássicos discordam.

**Modo contínuo:** essa penalidade opera sobre o Kelly base mesmo quando o motor já está em recovery Martingale Geométrico. A convergência adaptativa (seção 7.2) evita que a penalidade asfixie a recuperação financeira.

> **Gates defensivos removidos.** O Gate Assimétrico de Proteção (`validate_recovery_asymmetric_gate`), o Micro Noise Gate (`validate_micro_noise_gate`), o Filtro de Exaustão de Barreira Micro (`validate_micro_boundary_saturation_gate`) e o Veto de Inversão por Convicção DL foram eliminados. Não há mais SKIP por regime NEUTRO, chop de ADX, squeeze em random walk ou saturação de banda: o gate de qualidade é neutro e o boletamento é contínuo.

---

## 7. Recovery, sizing e persistência financeira

Esta seção documenta as diretrizes matemáticas que corrigem a **inanição por sizing desalinhado**: WINs operacionais com micro-stakes que zeravam o contador de perdas sem extinguir o drawdown real da sessão.

### 7.1 Filosofia de Recovery Financeiro Persistente

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

### 7.2 Waiver de Consensus Penalty em Recovery

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

### 7.3 Martingale Geométrico Puro sem teto (Kelly base × 2^n)

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

## 8. Execução

| Flag | Efeito |
|------|--------|
| `mandatory_trade_each_cycle: false` | Opera só com candidato acima do piso (modo seletivo) |
| `mandatory_trade_each_cycle: true` | Uma ordem por ciclo; qualidade como penalidade |
| `include_anchor_trades` | Inclui âncora nas ordens do cluster |
| `diversify_after_loss_margin` | Prefere símbolo alternativo quando scores são próximos |

Logs: `ord=` (ordem enviada) sempre igual a `dl=` (direção prevista pelo DL), pois não há mais inversão.

---

## 9. Risco e stop win por sessão

| Mecanismo | Papel |
|-----------|-------|
| Kelly fracionário | Sizing base com win rate dinâmico após amostras mínimas; compressão estática de 60% fora de recovery |
| Target Proximity Damping | Amortecimento linear da stake Kelly conforme `pnl_sessao` se aproxima de `target_win` (piso 0.40×) |
| Consensus Entropy Penalty | Defesa contra ruído CSPRNG (seção 6) |
| Penalty Smoothing | Convergência adaptativa em recovery (seção 7.2) |
| Recovery financeiro persistente | Estado de risco atrelado a `pending_total` (seção 7.1) |
| Martingale Geométrico sem teto | Recuperação exponencial `Kelly_base × 2^n` (seção 7.3) |
| Stop win por sessão ativa | `target_win = session_start_balance × compounding_rate_daily` (padrão 1%) |
| Stop loss | Desativado — sem reset por relógio nem disjuntor de perda diária |

### 9.1 Juros compostos e controle operacional

A meta segue a planilha de gerenciamento de juros compostos (`compounding_rate_daily: 0.01`):

| Evento | Comportamento |
|--------|---------------|
| Boot do processo | Captura saldo Deriv (ou `session_start_balance` em settings) como `session_start_balance` |
| Meta calculada | `target_win = session_start_balance × 0,01` (arredondada para baixo em centavos) |
| Durante a sessão | `pnl_sessao = current_balance - session_start_balance` |
| Meta atingida | `graceful_shutdown` — encerramento ordenado do motor |
| Restart manual | Nova sessão independente com novo saldo e nova meta de 1% |
| Mesmo dia civil | Múltiplas sessões isoladas permitidas — sem virada UTC/meia-noite |

Parâmetros em `risk_management.params`:

| Chave | Padrão | Função |
|-------|--------|--------|
| `compounding_enabled` | `true` | Ativa meta composta por sessão |
| `compounding_rate_daily` | `0.01` | Taxa de juros (1% sobre banca inicial) |
| `session_start_balance` | `null` | Override manual da banca inicial (senão usa saldo Deriv) |

Com `compounding_enabled: false`, o motor recorre ao alvo legado (`small_account_stop_win` / `large_account_stop_win_pct`).

### 9.2 Sizing defensivo de proximidade de alvo

Evita superexposição quando a sessão já capturou a maior parte do stop win de 1%:

1. **Kelly base comprimido** — `resolve_effective_kelly_fraction` aplica retenção de 40% (`fraction` de config `0.0035` → coeficiente `0.0012`), ancorando `Kelly_base` na faixa ~$10–$12 em vez de ~$31.
2. **Amortecimento dinâmico** — após o Kelly bruto, `apply_kelly_target_proximity_damping` multiplica a stake por `0.40 + 0.60 × remaining_target_pct`.
3. **Exemplo** — meta $101.20, Kelly bruto $45.56: com `pnl_sessao = 0` permanece $45.56×1.0 (já atenuado pela fração base); com 90% da meta (`pnl ≈ $91.08`) o fator cai para 0.46 (~$20.96).

Fora de recovery, este amortecimento define o `Kelly_base`. Em recovery, o Martingale Geométrico `Kelly_base × 2^n` opera sem amortecimento de proximidade.

Log de bootstrap: `SESSAO INICIADA | Alvo de 1%: $XX.XX | Stop Loss: DESATIVADO`.

---

## 10. Referências internas

- [arquitetura.md](arquitetura.md)
- [README.md](../README.md)
- [CHANGELOG.md](CHANGELOG.md)
