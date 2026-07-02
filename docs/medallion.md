# Metodologia quantitativa

O Aether Quantum Engine herda a postura **Medallion** no sentido operacional: o mercado é um **sistema de sinais ruidosos**, não uma narrativa macro discricionária. A implementação concentra-se nos índices **Drift** (`RDBEAR`, `RDBULL`) com **Deep Learning** e classificação binária Rise/Fall.

Para arquitetura de código, ver [`arquitetura.md`](arquitetura.md).

---

## 1. Princípios

| Princípio | No motor atual |
|-----------|----------------|
| Sinais, não histórias | Direção CALL/PUT por scoring numérico (DL + indicadores + trend) |
| Horizonte curto | Contexto DL **M15 (900 s)**; execução **M1 (60 s)**; label `ma_trend` |
| Qualidade adaptativa | Gate como penalidade em modo contínuo; veto seletivo quando configurado |
| Modelo pronto antes de operar | `FASE TREINO` suspende ordens até treino da sessão |
| Operação configurável | `mandatory_trade_each_cycle`: seletivo (`false`) ou contínuo (`true`) |
| Feedback real | Win rate live misturado em `val_accuracy`; retreino após loss |
| Defesa contra ruído CSPRNG | Consensus Entropy Penalty no Kelly; flip mean-reversion em exaustão |
| Persistência financeira | Recovery atrelado a `pending_loss`, não a WIN operacional isolado |
| Sobrevivência linear aditiva | Escada D'Alembert `Kelly + n×U` sem teto macro de stake |
| Meta por sessão ativa | Stop win de 1% composto sobre banca inicial; operador controla quantas sessões por dia |
| Sem disjuntor de perda | Stop loss interno desativado; recovery linear sem teto de drawdown imposto pelo motor |

---

## 2. Universo Drift

Índices sintéticos correlacionados no eixo de barreiras. Cada símbolo tem modelo DL independente com **34 features** e volatilidade calibrada ao alvo do índice.

| Símbolo | Papel típico |
|---------|----------------|
| `RDBULL` | Âncora padrão; referência de cluster |
| `RDBULL` / `RDBEAR` | Núcleo do cluster; bônus em recovery |
| `RDBEAR` / `RDBULL` | Pares de hedge para recovery |

Operação: contratos **RISE_FALL** (CALL = alta no período, PUT = queda).

---

## 3. Janela temporal de treino

## 3. Blindagem multi-timeframe

| Camada | Timeframe | Papel |
|--------|-----------|-------|
| Deep Learning / TCN | M15 (900 s) | Tensor `[1, 48, 34]` = 12 h de contexto macro |
| Orquestrador / contrato | M1 (60 s) | Ciclo a cada minuto; RISE_FALL de 60 s |
| Regimes universais | Indicadores M15 | Classificação estrutural (Trend, Compression, Climax) |
| Execução tática | M1 | Aplicação CALL/PUT na virada do minuto |

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
4. **Predição DL** — inferência Triton concorrente; `raw_prob` e indicadores calculados.
5. **Resolução direcional** — `execution_direction_resolver` + mean-reversion flip + veto de expansão.
6. **Gate de qualidade** — penalidade de score/edge (contínuo) ou veto (seletivo).
7. **Deploy** — `deploy_ok=false` bloqueia execução.
8. **Seleção** — `market_decision_score` entre candidatos elegíveis.
9. **Risco** — Kelly + Consensus Penalty; recovery financeiro persistente; escada D'Alembert linear; stop win por sessão ativa (1% composto).

Bloqueio absoluto **somente** para falhas técnicas. Conflitos de indicador ajustam direção, score e stake — não vetam participação no pool em modo contínuo.

Perfil em `config/settings.json`:

| Parâmetro | Valor | Função |
|-----------|-------|--------|
| `calibration.method` | auto | Platt, isotonic ou temperatura+Platt no holdout |
| `confidence_call_threshold` | 0.53 | Base de calibração CALL |
| `confidence_put_threshold` | 0.47 | Base de calibração PUT |
| `dynamic_threshold.enabled` | true | Thresholds flutuantes por volatilidade |
| `min_val_accuracy` | 0.60 | Piso de acurácia de validação |
| `min_edge_execute` | 0.04 | Edge base |
| `mandatory_min_trade_score` | 0.68 | Score mínimo modo normal |
| `mandatory_trade_each_cycle` | false | `true` = modo contínuo (uma ordem/ciclo) |
| `consensus_penalty_enabled` | true | Atenua Kelly quando ord diverge dos votos |
| `penalty_smoothing_factor` | 0.40 | Suavização convexa em recovery com trade_score > 0.68 |
| `mean_reversion_contraction_vol_ratio` | 0.80 | Limiar de contração para flip |
| `expansion_inversion_veto_vol_ratio` | 1.15 | Limiar de expansão para veto de inversão |

---

## 5. Scoring direcional

Pesos em `orchestrator.execution.direction_scoring`:

| Peso | Padrão | Sinal |
|------|--------|-------|
| `dl_raw_weight` | 0.45 | Probabilidade calibrada do modelo |
| `val_accuracy_weight` | 0.18 | Acurácia de validação |
| `trend_weight` | 0.15 | Alinhamento com tendência de mercado |
| `exhaustion_weight` | 0.12 | Mean-reversion em RSI/Keltner extremos |
| `indicator_regime_weight` | 0.10 | Hurst, ADX, vol_ratio, CMO |

O lado vencedor define `exec_direction`. `direction_inverted=true` quando difere de `dl_direction`.

### 5.1 Barramento de regimes CALL/PUT

Após o scoring composto, `UniversalRegimeEvaluator` classifica o candidato (34 features nas métricas) e aplica chaveamento direcional:

| Regime | Transição típica |
|--------|------------------|
| `TREND_EXPANSION` | `ord` = direção do DL (momentum) |
| `COMPRESSION_TRAP` | `ord` invertido quando preço esticado no canal lateral |
| `CLIMAX_EXHAUSTION` | `ord` contra o topo/fundo esticado do DL |
| `ENTROPIC_NOISE` | SKIP do ciclo (modo seletivo) ou direção por maior probabilidade calibrada com penalidade Kelly convexa máxima |

O barramento protege o Stop Win por sessão ao evitar falsos rompimentos em compressão e ruído CSPRNG.

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

**Modo contínuo:** essa penalidade opera sobre o Kelly base mesmo quando o motor já está em recovery D'Alembert. A convergência adaptativa (seção 7.2) evita que a penalidade asfixie a recuperação financeira.

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
| `consecutive_losses_linear` | Contador linear de clusters negativos — **memória operacional** de stress para escada D'Alembert |
| `dlambert_unit` (U) | Unidade aditiva capturada na primeira stake Kelly da sessão (ou override de config) |
| `recovery_financially_active` | Verdadeiro iff `pending_total > 0` |

#### Regra de persistência (implementação atual)

O motor **não utiliza reset cego** de `consecutive_losses_linear` baseado em WIN operacional isolado.

| Condição após liquidação | Comportamento |
|--------------------------|---------------|
| `cluster_profit < 0` | `consecutive_losses_linear += 1`; `pending_loss` incrementado |
| `cluster_profit ≥ 0` **e** `pending_total > 0` | WIN absorvido no drawdown; **`consecutive_losses_linear = max(1, n-1)`** (retração D'Alembert) |
| `cluster_profit ≥ 0` **e** `pending_total = 0` | Recovery financeiro extinto; reset de `consecutive_losses_linear` e `last_loss_stake` |

**Persistência de Drawdown:** o robô permanece em estado de Recovery (D'Alembert + sizing controlados + gates de convicção elevados) até que `pending_total` seja **financeiramente zerado** por retornos reais da API — não por um WIN simbólico que não cobre o buraco acumulado.

#### Implicação para gestão de cauda

- O **estado de risco** segue o **passivo financeiro** (`pending_loss`), não a contagem superficial de vitórias.
- Micro-WINs em recovery **amortizam** o drawdown e **retraem** a escada linear (`max(1, n-1)`), mas **não encerram** o regime de recovery prematuramente.
- Logs de auditoria: `RISK: WIN operacional`, `RISK: Lucro parcial`, `RISK: Recovery financeiro zerado` — cada um com `pend=$` e `pnl_sess=$`.

---

### 7.2 Regime Edge Sizing e waiver de Consensus Penalty em Recovery

#### Problema: penalidade convexa vs. inversão tática macro

O **Consensus Entropy Penalty** comprime `f*` quando a ordem diverge dos votos técnicos. Em recovery, quando o **Universal Regime Evaluator** inverte a direção do Deep Learning (`CLIMAX_EXHAUSTION` ou `COMPRESSION_TRAP` com `direction_inverted`), punir a stake por falta de consenso upstream anula a proteção macro — o sizing não pode contradizer a inversão tática.

#### Condições de waiver absoluto (`retention = 1.0`)

1. **Recovery ativo:** `pending_total > 0` **ou** `consecutive_losses_linear > 0`
2. **Qualquer** candidato do cluster em recovery que atenda **uma** das condições:
   - **Inversão de regime:** `universal_regime ∈ {CLIMAX_EXHAUSTION, COMPRESSION_TRAP}` **e** `direction_inverted = true`
   - **Votos unânimes alinhados:** `6×0` ou `0×6` na direção da ordem (M15)
   - **Convicção elevada:** `trade_score >= penalty_smoothing_trade_score_min` (padrão **0,68**)

Justificativa: com alinhamento direcional unânime em M15 ou convicção alta, o Kelly base não pode ser esmagado pela penalidade de entropia — o D'Alembert precisa operar com peso financeiro real em símbolos secundários do cluster (ex.: `RDBEAR`).

---

### 7.3 Escada D'Alembert com Amortization Booster (Kelly + unidade linear acelerada)

#### Problema: progressão multiplicativa vs. sobrevivência

A progressão multiplicativa (Martingale clássico) concentra risco geométrico: cada LOSS dobra a exposição e comprime a margem de erro da banca em sequências longas. O motor substituiu essa dinâmica por **sizing linear aditivo** inspirado em D'Alembert, ancorado na stake Kelly fracionária.

#### Fórmula de stake

| Estado | Stake |
|--------|-------|
| Normal (`pending_total = 0`) | Kelly fracionário (+ booster super-concordance se P≥0.75, 6×0, Hurst>0.55) |
| Recovery (`pending_total > 0`) | `Kelly_base + consecutive_losses_linear × U_eff` |

Onde:

| Termo | Significado |
|-------|-------------|
| `Kelly_base` | Stake Kelly após consensus penalty, scale e floor |
| `U` (`dlambert_unit`) | Primeira stake Kelly da sessão (ou `dlambert_unit_override` em config) |
| `U_eff` | Unidade acelerada pelo Amortization Booster quando há passivo pendente |
| `consecutive_losses_linear` | Contador linear de stress; +1 em LOSS de cluster |

#### Amortization Booster e Piso de Amortização Progressiva por Cluster

Quando `pending_total > 0`, a unidade base é escalonada pela profundidade do drawdown. Se o passivo excede **2% da banca** (`pending_total > bankroll × 0,02`), aplica-se um **piso de segurança** na unidade aditiva efetiva, **independente do símbolo** do cluster:

```
U_eff = max(U × 1,5, U × (1 + min(1,5, pending_total / (bankroll × 0,02))))
stake_raw = Kelly_base + consecutive_losses_linear × U_eff
```

Com drawdown até 2% da banca, mantém-se o escalonamento progressivo sem piso reforçado:

```
U_eff = U × (1 + min(1,5, pending_total / (bankroll × 0,02)))
```

**Exemplo:** banca $10.000, `pending_total = $400`, `U = $20`, `linear = 2`, `Kelly_base = $50`:

```
multiplier = 1 + min(1,5, 400/200) = 2,5
U_eff = 20 × 2,5 = $50
stake_raw = 50 + 2×50 = $150
```

#### Retração D'Alembert em WIN parcial

```
WIN parcial (pending_total > 0 após liquidação):
  consecutive_losses_linear = max(1, n - 1)

WIN total (pending_total = 0):
  consecutive_losses_linear = 0
  U preservado até nova sessão
```

A escada **sobe** uma unidade por LOSS e **desce** uma unidade por WIN parcial, nunca abaixo de 1 enquanto recovery ativo — evitando reset cego sem extinguir o drawdown.

#### Exemplo numérico

```
Sessão: U = $10 (Kelly capturado), Kelly_base = $8

Kelly puro:     stake = $8
LOSS #1:        linear=1 → stake = $8 + 1×$10 = $18
LOSS #2:        linear=2 → stake = $8 + 2×$10 = $28
WIN parcial:    linear=1 → stake = $8 + 1×$10 = $18
WIN total:      linear=0 → stake = $8 (Kelly puro)
```

#### Parâmetros de configuração (`risk_management.dlambert`)

| Parâmetro | Padrão | Função |
|-----------|--------|--------|
| `dlambert_enabled` | true | Ativa escada em recovery |
| `dlambert_unit_override` | null | Força U fixo (ignora captura Kelly) |
| `recovery_sizing_conviction` | 0,62 | Piso de convicção para sizing em recovery |
| `recovery_min_conviction` | 0,64 | Piso de convicção para entrada em recovery |
| `recovery_min_val_accuracy` | 0,62 | Piso de acurácia de validação |

#### Seleção de símbolo e Hurst em recovery

Complementar à escada linear:

- Ranking com diversificação e bônus em `RDBULL`/`RDBEAR`
- Trava Hurst N2+ (`recovery_hurst_gate`) — piso logarítmico de score
- `recovery_skip_counter` no Redis decai limiar Hurst em drawdown severo

---

## 8. Execução

| Flag | Efeito |
|------|--------|
| `mandatory_trade_each_cycle: false` | Opera só com candidato acima do piso (modo seletivo) |
| `mandatory_trade_each_cycle: true` | Uma ordem por ciclo; qualidade como penalidade |
| `include_anchor_trades` | Inclui âncora nas ordens do cluster |
| `diversify_after_loss_margin` | Prefere símbolo alternativo quando scores são próximos |

Logs: `ord=` (ordem enviada), `dl=` (direção prevista pelo DL), `inv` quando invertido.

---

## 9. Risco e stop win por sessão

| Mecanismo | Papel |
|-----------|-------|
| Kelly fracionário | Sizing base com win rate dinâmico após amostras mínimas; compressão estática de 60% fora de recovery |
| Target Proximity Damping | Amortecimento linear da stake Kelly conforme `pnl_sessao` se aproxima de `target_win` (piso 0.40×) |
| Consensus Entropy Penalty | Defesa contra ruído CSPRNG (seção 6) |
| Penalty Smoothing | Convergência adaptativa em recovery (seção 7.2) |
| Recovery financeiro persistente | Estado de risco atrelado a `pending_total` (seção 7.1) |
| Fatiamento progressivo | Sobrevivência geométrica N3+ (seção 7.3) |
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

1. **Kelly base comprimido** — `resolve_effective_kelly_fraction` aplica retenção de 40% (`fraction` de config `0.0035` → coeficiente `0.0012`), ancorando a unidade D'Alembert `U` na faixa ~$10–$12 em vez de ~$31.
2. **Amortecimento dinâmico** — após o Kelly bruto, `apply_kelly_target_proximity_damping` multiplica a stake por `0.40 + 0.60 × remaining_target_pct`.
3. **Exemplo** — meta $101.20, Kelly bruto $45.56: com `pnl_sessao = 0` permanece $45.56×1.0 (já atenuado pela fração base); com 90% da meta (`pnl ≈ $91.08`) o fator cai para 0.46 (~$20.96).

A escada D'Alembert continua aditiva sobre o `Kelly_base` já amortecido.

Log de bootstrap: `SESSAO INICIADA | Alvo de 1%: $XX.XX | Stop Loss: DESATIVADO`.

---

## 10. Referências internas

- [arquitetura.md](arquitetura.md)
- [README.md](../README.md)
- [CHANGELOG.md](CHANGELOG.md)
