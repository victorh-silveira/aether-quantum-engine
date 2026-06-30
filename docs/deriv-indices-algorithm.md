# Algoritmo de Índices Sintéticos da Deriv e Estratégia de Trading

Este documento detalha o funcionamento técnico dos índices sintéticos da Deriv (séries Volatility R_* e Range Break) e como o **Aether Quantum Engine** se posiciona estrategicamente para operá-los.

---

## 1. Como funciona o Algoritmo da Deriv

Os índices sintéticos da Deriv são gerados puramente por software e não são influenciados por eventos do mundo real ou notícias macroeconômicas.

```mermaid
flowchart TD
    Seed[Semente Aleatória Secreta] --> CSPRNG[Gerador Criptograficamente Seguro - CSPRNG]
    CSPRNG --> FloatSequence[Sequência Aleatória de Números]
    FloatSequence --> MathModel[Modelo Matemático da Série]
    VolCoef[Coeficiente de Volatilidade Constante] --> MathModel
    MathModel --> PriceFeed[Feed de Preços Contínuo / Ticks]
```

### 1.1 Gerador CSPRNG

O motor central de preços da Deriv utiliza um **Gerador de Números Pseudo-Aleatórios Criptograficamente Seguro (CSPRNG)**.

- **Segurança criptográfica**: computacionalmente inviável prever o próximo tick a partir de histórico observável.
- **Auditorias externas**: entidades independentes (e.g., eCOGRA) certificam aleatoriedade e integridade.

### 1.2 Parâmetros e Coeficientes de Volatilidade

Cada símbolo possui coeficiente de desvio padrão fixo (Volatility 10, 50, 75, 100, Range Break). O mercado opera 24/7 com comportamento estatístico consistente.

---

## 2. Estratégias de Trading para Índices Sintéticos

Como prever a saída exata do CSPRNG é impossível, o foco é **exploração de distorções estatísticas, padrões temporais e gerenciamento de risco rigoroso**.

### 2.1 Padrões de Momento e Tendência com TCN

O Aether utiliza **TCN** (padrão), **LSTM** ou **GRU** com conexões dilatadas ou recorrentes.

- **Por que TCN?** Processa janelas longas de lookback (48 barras × 180 s) identificando persistência de tendência ou exaustão.
- **34 features**: OHLC normalizado, indicadores técnicos, microestrutura de ticks e regime (Hurst, ADX, vol_ratio, CMO).
- **Inferência**: Triton gRPC concorrente (`TritonGrpcClient`) quando `infra.triton.enabled`; fallback local via TorchScript em cache.
- **Detecção de regime**: EMAs, inclinação, ADX e votos de trend (`dl_trend.py`) alimentam o scoring direcional.

### 2.2 Reversão à Média (Mean Reversion)

Desvios extremos tendem a retornar à média em ativos com volatilidade fixa.

- **Features de volatilidade**: `vol_ratio`, Hurst, variance ratio.
- **Exaustão no resolver**: RSI e Keltner extremos aplicam `exhaustion_flip` e `mean_reversion` no score CALL/PUT.
- **Flip dedicado** (`execution_direction_mean_reversion`): em exaustão com contração de vol (`vol_ratio < 0.80`), inverte a direção prevista pelo DL contra o consenso de exaustão.
- **Veto de expansão** (`execution_direction_expansion_veto`): com `vol_ratio > 1.15`, bloqueia inversão da ordem em relação ao DL e suaviza penalidade Kelly.

### 2.3 Gestão de Risco com Kelly Fracionário e Martingale Controlado

- **Fração de Kelly**: stake proporcional a `trade_score` calibrado e win rate live.
- **Consensus Entropy Penalty**: quando a ordem final diverge da maioria dos votos técnicos (`call_votes`/`put_votes`), aplica penalidade convexa em `f*` ponderando `di_diff`, `cmo` e afastamento do RSI; em baixo consenso, stake reduzida ao piso mínimo da API.
- **Gate de qualidade**: em modo seletivo, múltiplas camadas filtram execuções fracas; em modo contínuo, qualidade atua como penalidade de score/edge sem SKIP obrigatório.

### 2.4 Fases de Treinamento e Operação

- **FASE TREINO**: todos os símbolos retreinam ao menos uma vez por sessão (`session_trained`). Nenhuma ordem até concluir.
- **FASE OPERACAO seletiva** (`mandatory_trade_each_cycle: false`): opera quando o melhor candidato passa no gate (score ≥ 0.68 normal). Ciclos sem candidato elegível são pulados.
- **FASE OPERACAO contínua** (`mandatory_trade_each_cycle: true`): uma ordem por ciclo; qualidade como penalidade; fallback de entropia garante participação mínima.
- **Resolução direcional**: `execution_direction_resolver` combina probabilidade calibrada, trend, exaustão e regime; thresholds dinâmicos por `bb_width`/`atr_norm` ajustam convicção por índice (R_10 a R_100).
- **Deploy gate**: modelos com `deploy_ok=false` não entram no pool.

### 2.5 Perfil de qualidade atual

| Camada | Comportamento |
|--------|---------------|
| Bloqueio técnico | `data`, `predict_error`, `training`, `deploy_ok=false` |
| Calibração DL | Holdout ajusta Platt/isotonic; `calibrated_prob` alimenta scoring |
| Regime de vol | Compressão/estouro exige edge maior; regime direcional limpo relaxa pisos |
| Mean-reversion flip | Exaustão + `vol_ratio < 0.80` inverte direção DL |
| Expansão veto | `vol_ratio > 1.15` impede inversão ordem vs DL |
| Scoring direcional | Sempre CALL ou PUT quando tecnicamente válido |
| Gate de qualidade | Score ≥ 0.68, edge calibrado ≥ max(0.04, dynamic_min_edge), margem ≥ 0.05 (modo seletivo) |
| Inversão DL→exec | Exige score ≥ 0.74 (modo seletivo) |
| Modo normal | ADX ≥ 0.18 |
| Recovery | Pisos escalonados (0.64+) e martingale com convicção ≥ 0.64 |
| Kelly divergente | Consensus Entropy Penalty atenua stake quando ordem ≠ maioria dos votos |

---

## 3. Referências

- [arquitetura.md](arquitetura.md) — pipeline técnico
- [medallion.md](medallion.md) — princípios quant
- [infra-docker.md](infra-docker.md) — Triton, Redis, sanity estressado
- [README.md](../README.md) — execução
