# Metodologia quantitativa

O Aether Quantum Engine herda a postura **Medallion** no sentido operacional: o mercado é um **sistema de sinais ruidosos**, não uma narrativa macro discricionária. A implementação atual concentra-se no par sintético **Range Break** (`RDBULL` / `RDBEAR`) com **Deep Learning** online, não em clusters OTC transatlânticos nem LLM.

Para arquitetura de código, ver [`arquitetura.md`](arquitetura.md).

---

## 1. Princípios

| Princípio | No motor atual |
|-----------|----------------|
| Sinais, não histórias | Direção CALL/PUT a partir de features de preço e par; gating numérico |
| Horizonte curto | Velas 5m; contrato 1m; ciclo 300s |
| Regime e ruído | Meta-labels filtram movimentos irrelevantes; Brier e ECE no treino |
| Poucas operações de qualidade | `deploy_gate`, convicção mínima, pausa após losses |
| Feedback real | Win rate live misturado em `val_accuracy`; retreino após loss |

---

## 2. Par Range Break (RDBULL / RDBEAR)

Dois índices sintéticos correlacionados: o modelo usa **features de par** (spread, confirmação de direção) além de indicadores por símbolo.

| Símbolo | Papel típico |
|---------|----------------|
| `RDBULL` | Âncora configurável; referência de cluster |
| `RDBEAR` | Par complementar; seleção competitiva por `trade_score` |

Operação: contratos **RISE_FALL** (CALL = alta no período, PUT = queda).

---

## 3. Janela temporal de treino

Com `granularity: 300` (5 minutos):

| Conceito | Barras | Tempo aproximado |
|----------|--------|------------------|
| 1 dia civil | 288 | 24 h |
| Lookback do TCN | 96 (padrão) | 8 h de contexto por sequência |
| Validação holdout | 48 (padrão) | 4 h |
| Histórico usado no treino | `training_history_bars: 288` | 1 dia |

Configuração: `data_handler.history_bars`, `deep_learning.training_history_bars` ou `training_history_days`.

O motor recorta as últimas N barras antes de treinar e predizer (`slice_dl_price_window` em `dl_params.py`).

---

## 4. Camadas de decisão (qualidade)

Ordem lógica de uma entrada:

1. **Dados** — histórico suficiente (`gate_reason=data`).
2. **Modelo** — direção com margem (`direction_margin`).
3. **Deploy** — mini walk-forward pós-treino (`deploy`).
4. **Gating** — convicção, edge, val_acc, Brier, gap calibrado, saturação.
5. **Pós-loss** — ban temporário symbol+direção; cooldown de símbolo.
6. **Seleção** — melhor candidato entre RDBULL/RDBEAR.
7. **Risco** — Kelly ou martingale; stop win; stake mínima/máxima.

Perfil **qualidade** em `config/settings.json` (valores podem evoluir):

- `min_conviction_execute`, `min_edge_margin`, `min_val_accuracy`
- `max_val_brier_execute`, `max_calib_gap_execute`
- `recovery_gating` mais rígido que o modo normal
- `post_loss_ban_candles`, `session_max_losses_in_window`
- `deploy_gate`: `min_win_rate`, `max_brier`, `mini_bars` alinhado ao dia de histórico

---

## 5. Recovery e martingale

Após perda no cluster, `pending_loss` acumula valor a recuperar.

| Comportamento | Config |
|---------------|--------|
| Martingale ativo em recovery | `full_recovery_martingale`, `martingale_force_on_pending_loss` |
| Bloqueio por métricas DL | `martingale_dl_metrics_block` (Brier, `gate_reason`, `deploy_ok`) |
| Não repetir mesmo symbol+direção da última loss | `martingale_repeat_loss_blocked` |
| Convicção mínima em recovery (se force desligado) | `recovery_martingale_min_conviction` |

Stake de recovery cobre perda pendente + lucro alvo derivado do Kelly base.

---

## 6. Execução obrigatória e inversão

| Flag | Efeito |
|------|--------|
| `mandatory_trade_each_cycle` | Sempre envia ordem CALL ou PUT no ciclo (após seleção) |
| `invert_dl_direction` | Executa lado oposto ao previsto pelo DL (default: `false`) |
| `include_anchor_trades` | Inclui âncora nas ordens do cluster |

Logs: `ord=` (execução), `dl=` (previsto), `inv` quando invertido.

---

## 7. Risco e stop win

- **Kelly fracionário** com win rate dinâmico após amostras mínimas.
- **Stop win diário**: percentual da banca inicial (conta grande) ou valor fixo (conta pequena); novas entradas zeradas até o dia seguinte.
- **Sem martingale cego**: progressão só no modo recovery autorizado e sem bloqueio de gate.

---

## 8. Validação antes do live

1. Walk-forward: `dl_walkforward.py` por símbolo.
2. Verificar `deploy_ok`, win rate e Brier no relatório.
3. Apagar checkpoints antigos se a dimensão de features mudou.
4. Reiniciar o processo após alterar `settings.json`.

---

## 9. Legado Medallion (OTC / EURUSD / Gemini)

O repositório ainda contém scripts de backtest **Medallion** (`medallion_backtest.py`, HFT, coleta Gemini) para o universo histórico **frxEURUSD + índices OTC** com decisão LLM. Esse pipeline **não** alimenta o `Orchestrator` ao vivo na branch atual.

Use `dl_walkforward.py` como referência de validação para o par RD.

---

## 10. Referências internas

- [arquitetura.md](arquitetura.md)
- [README.md](../README.md)
- [CHANGELOG.md](CHANGELOG.md)
