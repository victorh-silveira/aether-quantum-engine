# Metodologia quantitativa

O Aether Quantum Engine herda a postura **Medallion** no sentido operacional: o mercado é um **sistema de sinais ruidosos**, não uma narrativa macro discricionária. A implementação concentra-se nos símbolos de **Range Break** (`R_10`, `R_25`, `R_50`, `R_75`, `R_100`) com **Deep Learning** online e classificação binária Rise/Fall.

Para arquitetura de código, ver [`arquitetura.md`](arquitetura.md).

---

## 1. Princípios

| Princípio | No motor atual |
|-----------|----------------|
| Sinais, não histórias | Direção CALL/PUT a partir de features numéricas (microestrutura, indicadores, Hurst, volatilidade) |
| Horizonte curto | Velas de 60 s; contrato 60 s (1 barra); label = close[t+1] > close[t] |
| Alta convicção | Opera apenas com `raw_prob >= 0.75` (CALL) ou `<= 0.25` (PUT); abstém no meio |
| Modelo pronto antes de operar | `FASE TREINO` suspende ordens até todos os modelos concluírem o treino da sessão |
| Operação seletiva | `mandatory_trade_each_cycle: false` — sem trade forçado por ciclo |
| Feedback real | Win rate live misturado em `val_accuracy`; retreino após loss |

---

## 2. Universo Range Break

Índices sintéticos correlacionados no eixo de barreiras. Cada símbolo tem modelo DL independente com features de volatilidade calibradas ao alvo do índice (ex.: R_75 → vol target 0.75).

| Símbolo | Papel típico |
|---------|----------------|
| `R_50` | Âncora padrão; referência de cluster |
| `R_10` / `R_100`, `R_25` / `R_75` | Pares de hedge para recovery |

Operação: contratos **RISE_FALL** (CALL = alta no período, PUT = queda).

---

## 3. Janela temporal de treino

Com `granularity: 60` (1 minuto) e `training_history_bars: 2880`:

| Conceito | Barras | Tempo aproximado |
|----------|--------|------------------|
| Histórico de treino | 2880 | 48 h |
| Lookback | 48 | 48 min de contexto por sequência |
| Validação holdout | 96 | 1 h 36 min |

Configuração: `data_handler.history_bars`, `deep_learning.training_history_bars`.

---

## 4. Camadas de decisão (qualidade)

Ordem lógica de uma entrada:

1. **Fase** — todos os modelos com treino da sessão concluído.
2. **Dados** — histórico suficiente (`gate_reason=data`).
3. **Treinamento** — modelo do símbolo treinado na sessão (`gate_reason=training`).
4. **Confiança** — `raw_prob >= 0.75` (CALL) ou `<= 0.25` (PUT); caso contrário abstém (`gate_reason=confidence`).
5. **Deploy** — mini walk-forward pós-treino (`deploy_ok=false` bloqueia execução).
6. **Val accuracy** — piso `min_val_accuracy: 0.53` (lucratividade mínima vs payout ~95%).
7. **Cooldown** — pausa por símbolo após losses.
8. **Seleção** — ranking de mercado entre símbolos elegíveis (`market_decision_score`).
9. **Risco** — Kelly ou martingale; stop win; stake mínima/máxima.

Perfil em `config/settings.json`:

- `confidence_call_threshold: 0.75`, `confidence_put_threshold: 0.25`
- `min_val_accuracy: 0.53`
- `deploy_gate`: `min_win_rate`, `max_brier`, `mini_bars`
- `mandatory_trade_each_cycle: false`

---

## 5. Recovery e martingale

Após perda no cluster, `pending_loss` acumula valor a recuperar.

| Comportamento | Implementação |
|---------------|-----------------|
| Martingale em recovery | Ativo sempre que `pending_loss > 0` |
| Fórmula de stake | `(perda pendente + seed × payout) / payout`, limitada por banca e `stake_max` |
| Seleção de símbolo | Ranking de mercado com diversificação (evita repetir símbolo perdedor) |
| Seleção de direção | Hedge no par Range (`recovery_hedge_target`) quando aplicável |
| Limites | `recovery_martingale_max_losses_per_symbol`, cooldown por símbolo |

---

## 6. Execução

| Flag | Efeito |
|------|--------|
| `mandatory_trade_each_cycle: false` | Opera somente quando DL atinge threshold de confiança; ciclo sem sinal forte é pulado |
| `include_anchor_trades` | Inclui âncora nas ordens do cluster |
| `diversify_after_loss_margin` | Prefere símbolo alternativo quando scores são próximos |

Direção de execução = direção inferida de `raw_prob` e threshold (CALL/PUT/abstém). Logs: `ord=` (ordem enviada), `dl=` (previsto pelo modelo).

---

## 7. Risco e stop win

- **Kelly fracionário** com win rate dinâmico após amostras mínimas.
- **Stop win diário**: percentual da banca inicial ou valor fixo.
- **Martingale** apenas em recovery, com recuperação integral da perda pendente.

---

## 8. Referências internas

- [arquitetura.md](arquitetura.md)
- [README.md](../README.md)
- [CHANGELOG.md](CHANGELOG.md)
