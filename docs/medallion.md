# Metodologia quantitativa

O Aether Quantum Engine herda a postura **Medallion** no sentido operacional: o mercado é um **sistema de sinais ruidosos**, não uma narrativa macro discricionária. A implementação atual concentra-se nos símbolos de **Range Break** (`R_10`, `R_25`, `R_50`, `R_75`, `R_100`) com **Deep Learning** online.

Para arquitetura de código, ver [`arquitetura.md`](arquitetura.md).

---

## 1. Princípios

| Princípio | No motor atual |
|-----------|----------------|
| Sinais, não histórias | Direção CALL/PUT a partir de features de preço e par; gating numérico |
| Horizonte curto | Velas de 1 minuto (`granularity: 60`); contrato 1m; ciclo de 60 s |
| Regime e ruído | Meta-labels filtram movimentos irrelevantes; Brier e ECE no treino |
| Modelo pronto antes de operar | `FASE TREINO` suspende ordens até todos os modelos concluírem o treino da sessão (`session_trained`) |
| Operação inteligente e seletiva | Até um trade por ciclo via ranking de mercado; ciclo pulado se nenhum candidato atinge `mandatory_min_trade_score` (0.53) |
| Feedback real | Win rate live misturado em `val_accuracy`; retreino após loss |

---

## 2. Universo Range Break

Índices sintéticos correlacionados no eixo de barreiras: o modelo usa **features de par** (spread, confirmação de direção) além de indicadores por símbolo.

| Símbolo | Papel típico |
|---------|----------------|
| `R_50` | Âncora padrão; referência de cluster |
| `R_10` / `R_100`, `R_25` / `R_75` | Pares de hedge para recovery |

Operação: contratos **RISE_FALL** (CALL = alta no período, PUT = queda).

---

## 3. Janela temporal de treino

Com `granularity: 60` (1 minuto) e `training_history_bars: 1440`:

| Conceito | Barras | Tempo aproximado |
|----------|--------|------------------|
| Histórico de treino | 1440 | 24 h |
| Lookback TCN | 32 | 32 min de contexto por sequência |
| Validação holdout | 60 | 1 h |

Configuração: `data_handler.history_bars`, `deep_learning.training_history_bars` ou `training_history_days`.

---

## 4. Camadas de decisão (qualidade)

Ordem lógica de uma entrada:

1. **Fase** — todos os modelos com treino da sessão concluído (`FASE TREINO` suspende a operação inteira).
2. **Dados** — histórico suficiente (`gate_reason=data`).
3. **Treinamento** — modelo do símbolo treinado na sessão (`gate_reason=training` nunca é forçado).
4. **Modelo** — direção com margem (`direction_margin`).
5. **Deploy** — mini walk-forward pós-treino (`deploy_ok=false` bloqueia execução e pool obrigatório).
6. **Gating** — convicção, edge, val_acc, Brier, gap calibrado, saturação.
7. **Regime** — alinhamento de momentum/RSI quando `require_regime_alignment` está ativo.
8. **Cooldown** — pausa por símbolo após losses (`symbol_loss_cooldown`, `session_pause_cycles`); símbolo em cooldown não entra nem no modo obrigatório.
9. **Seleção** — ranking de mercado entre símbolos elegíveis (`market_decision_score`) com piso `mandatory_min_trade_score`; em recovery, direção alinhada e diversificação de símbolo.
10. **Risco** — Kelly ou martingale; stop win; stake mínima/máxima.

Perfil em `config/settings.json`:

- `min_conviction_execute`, `min_edge_margin`, `min_val_accuracy`
- `max_val_brier_execute`, `max_calib_gap_execute`
- `recovery_gating` (modo recovery)
- `mandatory_min_trade_score` (piso de trade_score na execução obrigatória, padrão 0.53)
- `session_max_losses_in_window`, `session_pause_cycles`
- `deploy_gate`: `min_win_rate`, `max_brier`, `mini_bars`

---

## 5. Recovery e martingale

Após perda no cluster, `pending_loss` acumula valor a recuperar.

| Comportamento | Implementação |
|---------------|-----------------|
| Martingale em recovery | Ativo sempre que `pending_loss > 0` |
| Fórmula de stake | `(perda pendente + seed × payout) / payout`, limitada por banca e `stake_max` |
| Seleção de símbolo | Ranking de mercado com bônus de diversificação (evita repetir o símbolo perdedor) e núcleo `R_75`/`R_50` |
| Seleção de direção | Alinhada ao último loss com pisos de qualidade; fallback por hedge no par (`execution_symbols_recovery`) |
| Gating DL em recovery | `recovery_gating` + `recovery_allow_bypass` |
| Limites | `recovery_martingale_max_losses_per_symbol` (símbolo sai do pool de recovery após sequência de losses) e cooldown por símbolo (`symbol_loss_cooldown_candles`) impedem reentrada cega no mesmo par |

---

## 6. Execução

| Flag | Efeito |
|------|--------|
| `mandatory_trade_each_cycle` | Tenta uma ordem por ciclo na fase de operação, escolhida por ranking de mercado; pula o ciclo se nenhum candidato atinge `mandatory_min_trade_score` |
| `mandatory_min_trade_score` | Piso mínimo de `trade_score` em todos os caminhos obrigatórios (pool, ranking, fallback) |
| `include_anchor_trades` | Inclui âncora nas ordens do cluster |
| `diversify_after_loss_margin` | Prefere símbolo alternativo quando scores são próximos |

Direção de execução = direção DL refinada por `resolve_market_direction`: com convicção bruta fraca, extremos estatísticos da vela (`sma_z`) aplicam reversão à média. Logs: `ord=` (ordem enviada), `dl=` (previsto pelo modelo); `direction_inverted` marca divergência.

---

## 7. Risco e stop win

- **Kelly fracionário** com win rate dinâmico após amostras mínimas.
- **Stop win diário**: percentual da banca inicial (conta grande) ou valor fixo (conta pequena).
- **Martingale** apenas em recovery, com recuperação integral da perda pendente.

---

## 8. Referências internas

- [arquitetura.md](arquitetura.md)
- [README.md](../README.md)
- [CHANGELOG.md](CHANGELOG.md)
