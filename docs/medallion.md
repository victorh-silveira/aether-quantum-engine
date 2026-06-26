# Metodologia quantitativa

O Aether Quantum Engine herda a postura **Medallion** no sentido operacional: o mercado é um **sistema de sinais ruidosos**, não uma narrativa macro discricionária. A implementação concentra-se nos símbolos de **Range Break** (`R_10`, `R_25`, `R_50`, `R_75`, `R_100`) com **Deep Learning** e classificação binária Rise/Fall.

Para arquitetura de código, ver [`arquitetura.md`](arquitetura.md).

---

## 1. Princípios

| Princípio | No motor atual |
|-----------|----------------|
| Sinais, não histórias | Direção CALL/PUT por scoring numérico (DL + indicadores + trend) |
| Horizonte curto | Velas de 180 s; contrato 180 s; label `ma_trend` |
| Qualidade > quantidade | Gate pós-resolução; ciclo pulado sem candidato forte |
| Modelo pronto antes de operar | `FASE TREINO` suspende ordens até treino da sessão |
| Operação seletiva | `mandatory_trade_each_cycle: false` |
| Feedback real | Win rate live misturado em `val_accuracy`; retreino após loss |

---

## 2. Universo Range Break

Índices sintéticos correlacionados no eixo de barreiras. Cada símbolo tem modelo DL independente com features de volatilidade calibradas ao alvo do índice.

| Símbolo | Papel típico |
|---------|----------------|
| `R_10` | Âncora padrão; referência de cluster |
| `R_50` / `R_75` | Núcleo do cluster; bonus em recovery |
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
4. **Predição DL** — `raw_prob` e indicadores calculados; `execute=True` se técnico OK.
5. **Resolução direcional** — `execution_direction_resolver`: CALL ou PUT por score composto.
6. **Gate de qualidade** — score, edge, margem direcional, ADX, inversão.
7. **Deploy** — `deploy_ok=false` bloqueia execução.
8. **Seleção** — `market_decision_score` entre candidatos elegíveis.
9. **Risco** — Kelly ou martingale; stop win; stake mínima/máxima.

Bloqueio absoluto **somente** para falhas técnicas. Conflitos de indicador (trend, exaustão, regime) **ajustam** direção e score, não vetam participação no pool.

Perfil em `config/settings.json`:

| Parâmetro | Valor | Função |
|-----------|-------|--------|
| `confidence_call_threshold` | 0.53 | Referência de calibração CALL |
| `confidence_put_threshold` | 0.47 | Referência de calibração PUT |
| `min_val_accuracy` | 0.60 | Piso de acurácia de validação |
| `min_edge_execute` | 0.04 | Edge mínimo para execução |
| `mandatory_min_trade_score` | 0.68 | Score mínimo modo normal |
| `recovery_min_trade_score` | 0.64 | Score mínimo recovery |
| `quality_gate.inverted_min_score` | 0.74 | Score mínimo com inversão DL→exec |
| `quality_gate.min_adx_normal` | 0.18 | ADX mínimo fora de recovery |
| `mandatory_trade_each_cycle` | false | Sem trade forçado por ciclo |

---

## 5. Scoring direcional

Pesos em `orchestrator.execution.direction_scoring`:

| Peso | Padrão | Sinal |
|------|--------|-------|
| `dl_raw_weight` | 0.45 | Probabilidade bruta do modelo |
| `val_accuracy_weight` | 0.18 | Acurácia de validação / flip em val baixa |
| `trend_weight` | 0.15 | Alinhamento com tendência de mercado |
| `exhaustion_weight` | 0.12 | Mean-reversion em RSI/Keltner extremos |
| `indicator_regime_weight` | 0.10 | Hurst, ADX, vol_ratio, CMO |

O lado vencedor define `exec_direction`. `direction_inverted=true` quando difere de `dl_direction`.

---

## 6. Recovery e martingale

Após perda no cluster, `pending_loss` acumula valor a recuperar.

| Comportamento | Implementação |
|---------------|---------------|
| Martingale em recovery | Ativo quando `pending_loss > 0` |
| Fórmula de stake | `(perda pendente + alvo) / payout`, limitada por banca |
| Seleção de símbolo | Ranking com diversificação e bonus em `R_50`/`R_75` |
| Convicção mínima | `recovery_martingale_min_conviction: 0.64` |
| Val accuracy | `recovery_min_val_accuracy: 0.62` |
| Escalonamento | Pisos sobem com `consecutive_losses` |
| Limites | `recovery_martingale_max_losses_per_symbol`, cooldown |

---

## 7. Execução

| Flag | Efeito |
|------|--------|
| `mandatory_trade_each_cycle: false` | Opera só com candidato acima do piso de qualidade |
| `include_anchor_trades` | Inclui âncora nas ordens do cluster |
| `diversify_after_loss_margin` | Prefere símbolo alternativo quando scores são próximos |
| `recovery_flip_direction_after_loss` | false (inversão via scoring, não flip binário) |

Logs: `ord=` (ordem enviada), `dl=` (direção prevista pelo DL), `inv` quando invertido.

---

## 8. Risco e stop win

- **Kelly fracionário** com win rate dinâmico após amostras mínimas.
- **Stop win diário**: percentual da banca inicial ou valor fixo.
- **Martingale** apenas em recovery, com pisos de convicção elevados.

---

## 9. Referências internas

- [arquitetura.md](arquitetura.md)
- [README.md](../README.md)
- [CHANGELOG.md](CHANGELOG.md)
