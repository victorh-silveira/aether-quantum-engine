# Metodologia quantitativa

O Aether Quantum Engine herda a postura **Medallion** no sentido operacional: o mercado é um **sistema de sinais ruidosos**, não uma narrativa macro discricionária. A implementação atual concentra-se nos símbolos de **Range Break** (`R_10`, `R_25`, `R_50`, `R_75`, `R_100`) com **Deep Learning** online.

Para arquitetura de código, ver [`arquitetura.md`](arquitetura.md).

---

## 1. Princípios

| Princípio | No motor atual |
|-----------|----------------|
| Sinais, não histórias | Direção CALL/PUT a partir de features de preço e par; gating numérico |
| Horizonte curto | Velas configuráveis (`granularity`); contrato 1m; ciclo 300s |
| Regime e ruído | Meta-labels filtram movimentos irrelevantes; Brier e ECE no treino |
| Poucas operações de qualidade | `deploy_gate`, convicção mínima, pausa após losses |
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

1. **Dados** — histórico suficiente (`gate_reason=data`).
2. **Modelo** — direção com margem (`direction_margin`).
3. **Deploy** — mini walk-forward pós-treino (`deploy`).
4. **Gating** — convicção, edge, val_acc, Brier, gap calibrado, saturação.
5. **Regime** — alinhamento de momentum/RSI quando `require_regime_alignment` está ativo.
6. **Cooldown** — pausa por símbolo após losses (`symbol_loss_cooldown`, `session_pause_cycles`).
7. **Seleção** — melhor candidato entre símbolos elegíveis; em recovery, hedge no par.
8. **Risco** — Kelly ou martingale; stop win; stake mínima/máxima.

Perfil em `config/settings.json`:

- `min_conviction_execute`, `min_edge_margin`, `min_val_accuracy`
- `max_val_brier_execute`, `max_calib_gap_execute`
- `recovery_gating` (modo recovery)
- `session_max_losses_in_window`, `session_pause_cycles`
- `deploy_gate`: `min_win_rate`, `max_brier`, `mini_bars`

---

## 5. Recovery e martingale

Após perda no cluster, `pending_loss` acumula valor a recuperar.

| Comportamento | Implementação |
|---------------|-----------------|
| Martingale em recovery | Ativo sempre que `pending_loss > 0` |
| Fórmula de stake | `(perda pendente + seed × payout) / payout`, limitada por banca e `stake_max` |
| Seleção de direção | Hedge no par oposto à última loss (`execution_symbols_recovery`) |
| Gating DL em recovery | `recovery_gating` + `recovery_allow_bypass` |

Não há teto de passos de martingale nem bloqueio por repetir symbol+direção da última loss.

---

## 6. Execução

| Flag | Efeito |
|------|--------|
| `mandatory_trade_each_cycle` | Envia ordem a cada ciclo elegível (stake cap para sinais fracos) |
| `include_anchor_trades` | Inclui âncora nas ordens do cluster |
| `diversify_after_loss_margin` | Prefere símbolo alternativo quando scores são próximos |

Direção de execução = direção prevista pelo DL. Logs: `ord=` (ordem enviada), `dl=` (previsto pelo modelo).

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
