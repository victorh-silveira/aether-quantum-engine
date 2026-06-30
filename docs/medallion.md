# Metodologia quantitativa

O Aether Quantum Engine herda a postura **Medallion** no sentido operacional: o mercado é um **sistema de sinais ruidosos**, não uma narrativa macro discricionária. A implementação concentra-se nos símbolos de **Range Break** (`R_10`, `R_25`, `R_50`, `R_75`, `R_100`) com **Deep Learning** e classificação binária Rise/Fall.

Para arquitetura de código, ver [`arquitetura.md`](arquitetura.md).

---

## 1. Princípios

| Princípio | No motor atual |
|-----------|----------------|
| Sinais, não histórias | Direção CALL/PUT por scoring numérico (DL + indicadores + trend) |
| Horizonte curto | Velas de 180 s; contrato 180 s; label `ma_trend` |
| Qualidade adaptativa | Gate como penalidade em modo contínuo; veto seletivo quando configurado |
| Modelo pronto antes de operar | `FASE TREINO` suspende ordens até treino da sessão |
| Operação configurável | `mandatory_trade_each_cycle`: seletivo (`false`) ou contínuo (`true`) |
| Feedback real | Win rate live misturado em `val_accuracy`; retreino após loss |
| Defesa contra ruído CSPRNG | Consensus Entropy Penalty no Kelly; flip mean-reversion em exaustão |

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

Com `granularity: 180` (3 minutos) e `training_history_bars: 25920`:

| Conceito | Barras | Tempo aproximado |
|----------|--------|------------------|
| Histórico de treino | 25920 | ~54 dias |
| Lookback | 48 | 24 h de contexto por sequência |
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
9. **Risco** — Kelly com Consensus Entropy Penalty; martingale; stop win.

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

## 6. Kelly e Consensus Entropy Penalty

Quando a ordem final (`order_direction`) diverge da maioria dos votos técnicos (`call_votes`/`put_votes`):

- Calcula taxa de concordância e divergência.
- Aplica penalidade **convexa** (`divergence^exponent`) ponderada por `di_diff`, `cmo` e afastamento do RSI em sentido oposto.
- Reduz `f*` proporcionalmente; em baixo consenso (`consensus_min_retention`), stake forçada ao piso mínimo da API.

Objetivo: preservar capital contra ruído do CSPRNG quando o DL e os indicadores clássicos discordam.

---

## 7. Recovery e martingale

Após perda no cluster, `pending_loss` acumula valor a recuperar.

| Comportamento | Implementação |
|---------------|---------------|
| Martingale em recovery | Ativo quando `pending_loss > 0` |
| Fórmula de stake | `(perda pendente + alvo) / payout`, limitada por banca |
| Seleção de símbolo | Ranking com diversificação e bônus em `R_50`/`R_75` |
| Convicção mínima | `recovery_martingale_min_conviction: 0.64` |
| Val accuracy | `recovery_min_val_accuracy: 0.62` |
| Hurst N2+ | Piso logarítmico; `recovery_skip_counter` decai limiar no Redis |

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

## 9. Risco e stop win

- **Kelly fracionário** com win rate dinâmico após amostras mínimas.
- **Consensus Entropy Penalty** em divergência ordem vs votos.
- **Stop win diário**: percentual da banca inicial ou valor fixo.
- **Martingale** apenas em recovery, com pisos de convicção elevados.

---

## 10. Referências internas

- [arquitetura.md](arquitetura.md)
- [README.md](../README.md)
- [CHANGELOG.md](CHANGELOG.md)
