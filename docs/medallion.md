# Metodologia quantitativa

O Aether Quantum Engine herda a postura **Medallion** no sentido operacional: o mercado é um **sistema de sinais ruidosos**, não uma narrativa macro discricionária. A implementação concentra-se nos símbolos de **Range Break** (`R_10`, `R_25`, `R_50`, `R_75`, `R_100`) com **Deep Learning** e classificação binária Rise/Fall.

Para arquitetura de código, ver [`arquitetura.md`](arquitetura.md).

---

## 1. Princípios

| Princípio | No motor atual |
|-----------|----------------|
| Sinais, não histórias | Direção CALL/PUT por scoring numérico (DL + indicadores + trend) |
| Horizonte curto | Velas **M5 (300 s)**; contrato **300 s**; label `ma_trend` |
| Qualidade adaptativa | Gate como penalidade em modo contínuo; veto seletivo quando configurado |
| Modelo pronto antes de operar | `FASE TREINO` suspende ordens até treino da sessão |
| Operação configurável | `mandatory_trade_each_cycle`: seletivo (`false`) ou contínuo (`true`) |
| Feedback real | Win rate live misturado em `val_accuracy`; retreino após loss |
| Defesa contra ruído CSPRNG | Consensus Entropy Penalty no Kelly; flip mean-reversion em exaustão |
| Persistência financeira | Recovery atrelado a `pending_loss`, não a WIN operacional isolado |
| Sobrevivência geométrica | Fatiamento progressivo N3+ e CAP 4% da banca em Martingale |
| Meta por sessão ativa | Stop win de 1% composto sobre banca inicial; operador controla quantas sessões por dia |
| Sem disjuntor de perda | Stop loss interno desativado; Martingale sem teto de drawdown imposto pelo motor |

---

## 2. Universo Range Break

Índices sintéticos correlacionados no eixo de barreiras. Cada símbolo tem modelo DL independente com **34 features** e volatilidade calibrada ao alvo do índice.

| Símbolo | Papel típico |
|---------|----------------|
| `R_10` | Âncora padrão; referência de cluster |
| `R_50` / `R_75` | Núcleo do cluster; bônus em recovery |
| `R_10` / `R_100`, `R_25` / `R_75` | Pares de hedge para recovery |

Operação: contratos **RISE_FALL** (CALL = alta no período, PUT = queda).

---

## 3. Janela temporal de treino

Com `granularity: 300` (5 minutos / M5) e `training_history_bars: 15552`:

| Conceito | Barras | Tempo aproximado |
|----------|--------|------------------|
| Histórico de treino | 15552 | ~54 dias |
| Lookback | 48 | **4 h** de contexto por sequência |
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
9. **Risco** — Kelly + Consensus Penalty; recovery financeiro persistente; fatiamento Martingale; stop win por sessão ativa (1% composto).

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
| `penalty_smoothing_factor` | 0.40 | Suavização convexa em recovery com trade_score > 0.70 |
| `martingale_hard_cap_bankroll_pct` | 0.04 | CAP crítico 4% da banca em Martingale |
| `martingale_safety_losses_min` | 3 | Ativa fatiamento progressivo a partir de N3 |
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

**Modo contínuo:** essa penalidade opera sobre o Kelly base mesmo quando o motor já está em recovery Martingale. A convergência adaptativa (seção 7.2) evita que a penalidade asfixie a recuperação financeira.

---

## 7. Recovery, sizing e persistência financeira

Esta seção documenta as diretrizes matemáticas que corrigem a **inanição por sizing desalinhado**: WINs operacionais com micro-stakes que zeravam o contador de perdas sem extinguir o drawdown real da sessão.

### 7.1 Filosofia de Recovery Financeiro Persistente

#### Problema: reset cego vs. realidade financeira

Em execução contínua (`mandatory_trade_each_cycle: true`), o motor pode registrar um **WIN operacional** (P&L positivo no contrato liquidado) enquanto o **saldo acumulado da sessão** (`total_session_profit`) permanece negativo e o **drawdown pendente** (`pending_loss`) ainda carrega valor a recuperar.

O critério legado — resetar `consecutive_losses` a zero após qualquer cluster com P&L ≥ 0 — tratava **resultado operacional isolado** como **recuperação financeira completa**. Isso gerava assimetria negativa:

```
Ciclo 1: LOSS  -$10  →  pending_loss = $10,  consecutive_losses = 1,  MARTINGALE
Ciclo 2: WIN   +$3   →  pending_loss = $7,   consecutive_losses = 0   ← reset cego
Ciclo 3: LOSS  -$12  →  nova linha de perda sem memória de recovery
```

O robô voltava ao Kelly fracionário com stakes micro (~$8), enquanto a sessão continuava no vermelho — **inanição por sizing desalinhado**.

#### Definições quantitativas

| Variável | Significado |
|----------|-------------|
| `pending_loss[s]` | Drawdown financeiro pendente por símbolo `s`, acumulado após losses e reduzido por wins via `apply_win_to_pending_loss` |
| `pending_total` | `Σ pending_loss[s]` — critério único de recovery financeiro ativo |
| `total_session_profit` | P&L acumulado real da sessão (soma de todos os contratos liquidados pela API) |
| `consecutive_losses` | Contador de clusters negativos consecutivos — **memória operacional** de stress |
| `recovery_financially_active` | Verdadeiro iff `pending_total > 0` |

#### Regra de persistência (implementação atual)

O motor **não utiliza mais reset cego** de `consecutive_losses` baseado em WIN operacional isolado.

| Condição após liquidação | Comportamento |
|--------------------------|---------------|
| `cluster_profit < 0` | `consecutive_losses += 1`; `pending_loss` incrementado |
| `cluster_profit ≥ 0` **e** `pending_total > 0` | WIN absorvido no drawdown; **`consecutive_losses` mantido**; modo MARTINGALE preservado |
| `cluster_profit ≥ 0` **e** `pending_total = 0` | Recovery financeiro extinto; reset de `consecutive_losses`, `last_martingale_stake` e `last_loss_stake` |

**Persistência de Drawdown:** o robô permanece em estado de Recovery (Martingale + sizing controlados + gates de convicção elevados) até que `pending_total` seja **financeiramente zerado** por retornos reais da API — não por um WIN simbólico que não cobre o buraco acumulado.

#### Implicação para gestão de cauda

- O **estado de risco** segue o **passivo financeiro** (`pending_loss`), não a contagem superficial de vitórias.
- Micro-WINs em recovery **amortizam** o drawdown, mas **não encerram** o regime Martingale prematuramente.
- Logs de auditoria: `RISK: WIN operacional`, `RISK: Lucro parcial`, `RISK: Recovery financeiro zerado` — cada um com `pend=$` e `pnl_sess=$`.

---

### 7.2 Convergência Adaptativa do Kelly em Recovery (Penalty Smoothing Factor)

#### Problema: penalidade convexa vs. eficiência de recuperação

O **Consensus Entropy Penalty** (seção 6) comprime `f*` quando a ordem diverge dos votos técnicos. Em modo contínuo, com `Acc ≈ 0,69` e consenso fraco, a penalidade convexa empurrava stakes para o piso mínimo ($1,00) **mesmo durante recovery Martingale** — impossibilitando a extração matemática do drawdown pendente.

#### Condições de ativação

O **Penalty Smoothing Factor** (`penalty_smoothing_factor`, padrão **0,40**) aplica-se quando **todas** as condições abaixo são verdadeiras:

1. **Recovery ativo:** `pending_total > 0` **ou** `consecutive_losses > 0`
2. **Sinal robusto:** `trade_score > penalty_smoothing_trade_score_min` (padrão **0,70**)
3. **Penalidade em vigor:** `retention_raw < 1,0` (ordem diverge do consenso)

#### Fórmula de suavização

```
cut          = 1 - retention_raw
retention*   = min(1, retention_raw + cut × penalty_smoothing_factor)
f*_efetivo   = f* × retention*
```

Com `penalty_smoothing_factor = 0,40`, **40% do corte convexa é devolvido** à stake — a penalidade efetiva cai de `cut` para `cut × (1 - 0,40) = cut × 0,60`.

**Exemplo numérico:**

| Grandeza | Valor |
|----------|-------|
| `retention_raw` | 0,50 (corte de 50%) |
| `penalty_smoothing_factor` | 0,40 |
| `retention*` | 0,50 + 0,50 × 0,40 = **0,70** |
| Efeito | Stake Kelly recupera 20 p.p. de retenção |

#### Fronteira de segurança: CAP 4%

A suavização **nunca** eleva a stake acima do teto Martingale:

```
stake_final ≤ bankroll × martingale_hard_cap_bankroll_pct   (padrão 0,04 = 4%)
```

A convergência adaptativa aumenta a **eficiência** da recuperação dentro do envelope de sobrevivência geométrica — não o risco absoluto sobre a banca global.

#### Objetivo quantitativo

Permitir que entradas com `trade_score > 0,70` em recovery tenham stake **matematicamente suficiente** para convergir `pending_total → 0` em horizonte finito, sem sacrificar a defesa contra divergência ordem-vs-votos em regime normal (fora de recovery, `retention*` = `retention_raw`).

---

### 7.3 Dinâmica de Fatiamento Progressivo do Martingale

#### Problema: concentração de risco em sequências longas

Recuperar `pending_total` integralmente em **um único ciclo** de 60 s expõe a banca a **risco geométrico concentrado** — especialmente após N3+ perdas consecutivas em execução contínua, onde cada ciclo força participação.

O fatiamento progressivo **fragmenta** o passivo pendente em parcelas distribuídas por múltiplos ciclos futuros, preservando a direção de recuperação sem apostar a sobrevivência em uma tacada única.

#### Tabela de fatiamento (`martingale_progressive_slice_cycles`)

Parâmetro de ativação: `martingale_safety_losses_min` (padrão **3**).

| `consecutive_losses` | Divisor `slice_cycles` | Interpretação |
|----------------------|------------------------|---------------|
| N < 3 | **1** | Recovery integral em um ciclo (comportamento clássico) |
| N = 3 | **2** (`martingale_progressive_slice_at_3`) | Metade do `pending_total` efetivo por ciclo |
| N ≥ 4 | **3** (`martingale_progressive_slice_at_4plus`) | Um terço do `pending_total` efetivo por ciclo |

#### Fórmula de stake fragmentada

```
effective_loss = pending_total × step_frac / slice_cycles
profit_target  = seed × payout × martingale_target_fraction
stake_raw      = (effective_loss + profit_target) / payout
stake_final    = min(stake_raw, bankroll × martingale_hard_cap_bankroll_pct)
```

Onde:

| Termo | Significado |
|-------|-------------|
| `step_frac` | Fração de recuperação por ciclo (ajustada por vol: defer 50% se `vol_ratio > 1,10` em N2+) |
| `seed` | `max(last_loss_stake, kelly_base, stake_min)` — referência de progressão |
| `slice_cycles` | Divisor de fatiamento conforme tabela acima |

#### Assimetria de sobrevivência geométrica

Em execução contínua, a sequência N1 → N2 → N3+ comprime geometricamente a margem de erro da banca. O fatiamento introduz **convexidade defensiva**:

```
Sem fatiamento (N=4, pending=$30):
  stake ≈ $30/payout  →  risco de ruína elevado se LOSS

Com fatiamento (slice_cycles=3):
  effective_loss = $30 × step_frac / 3  →  ~$10/ciclo
  3 ciclos de WIN parcial convergem pending sem concentração
```

A combinação **fatiamento + CAP 4% + vol-adjust defer** forma um envelope tridimensional de sobrevivência:

1. **Horizontal (temporal):** parcelas em 2–3 ciclos
2. **Vertical (banca):** teto 4% por tacada
3. **Regime (volatilidade):** `step_frac` reduzido em expansão (`vol_ratio > 1,10`)

#### Parâmetros de configuração

| Parâmetro | Padrão | Função |
|-----------|--------|--------|
| `martingale_safety_losses_min` | 3 | Ativa fatiamento a partir de N3 |
| `martingale_progressive_slice_at_3` | 2 | Divisor em N = 3 |
| `martingale_progressive_slice_at_4plus` | 3 | Divisor em N ≥ 4 |
| `martingale_hard_cap_bankroll_pct` | 0,04 | CAP crítico 4% da banca |
| `recovery_martingale_min_conviction` | 0,64 | Piso de convicção em recovery |
| `recovery_min_val_accuracy` | 0,62 | Piso de acurácia de validação |

#### Seleção de símbolo e Hurst em recovery

Complementar ao fatiamento:

- Ranking com diversificação e bônus em `R_50`/`R_75`
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
| Kelly fracionário | Sizing base com win rate dinâmico após amostras mínimas |
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

Log de bootstrap: `SESSAO INICIADA | Alvo de 1%: $XX.XX | Stop Loss: DESATIVADO`.

---

## 10. Referências internas

- [arquitetura.md](arquitetura.md)
- [README.md](../README.md)
- [CHANGELOG.md](CHANGELOG.md)
