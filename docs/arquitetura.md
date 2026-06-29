# Arquitetura — Aether Quantum Engine

Motor assíncrono para trading na Deriv com decisão exclusiva por **Deep Learning** (TCN, LSTM ou GRU) nos símbolos **Range Break** (`R_10`, `R_25`, `R_50`, `R_75`, `R_100`). A metodologia de negócio quantitativa está em [`medallion.md`](medallion.md); este documento descreve o software.

---

## 1. Visão geral

| Aspecto | Valor atual (`config/settings.json`) |
|---------|--------------------------------------|
| Símbolos | `R_10`, `R_25`, `R_50`, `R_75`, `R_100` (âncora `R_10`) |
| Granularidade OHLC | 180 s (`data_handler.granularity`) |
| Histórico para treino | 25920 barras (`training_history_bars`) |
| Lookback | 48 barras por sequência |
| Contrato | `RISE_FALL`, duração 180 s |
| Ciclo do orquestrador | 60 s (`cycle_interval_seconds`) |
| Decisão | `collect_deep_learning_decisions` |
| Fases | `FASE TREINO` → `FASE OPERACAO` |
| Execução | Seletiva com gate de qualidade; `mandatory_trade_each_cycle: false` |

O mercado é tratado como **série temporal ruidosa**: o modelo estima probabilidade de alta; um **motor de direção** composto e um **gate de qualidade** decidem CALL, PUT ou skip do ciclo.

---

## 2. Pipeline de dados

```mermaid
flowchart LR
  subgraph ingestao
    WS[WebSocketManager]
    SH[StreamHandler]
    TB[TickBuffer]
  end
  subgraph dl
    FEAT[dl_features 19D]
    MODEL[TCN ou LSTM/GRU]
    PRED[dl_predict]
  end
  subgraph direcao
    ENT[execution_entropy_adaptive]
    RES[execution_direction_resolver]
    EXH[execution_exhaustion_hard_gate]
    QG[execution_quality_gate]
  end
  subgraph exec
    COL[execution_collect]
    SEL[execution_symbols]
    EM[ExecutionManager]
    TH[TradeHandler]
  end
  subgraph pos
    ST[settlement_*]
    RM[RiskManager]
    PM[Redis StateStore]
    TS[TimescaleDB]
    MO[MinIO]
  end
  WS --> SH
  SH --> TB
  SH --> FEAT --> MODEL --> PRED --> ENT --> RES --> EXH --> QG --> COL --> SEL --> EM --> TH
  TH --> ST --> RM
  ST --> PM
  ST --> TS
  MODEL --> MO
```

### 2.4 Infraestrutura hibrida

Com `infra.enabled: true`, o motor valida Redis, TimescaleDB e MinIO em `localhost` antes do WebSocket (fail-fast). Estado de risco e sessao persistem em Redis via pipeline atomico (`redis_state_pipeline.write_state_bundle`); ticks e barras vao para Timescale; checkpoints DL sincronizam com MinIO mantendo cache local em `data/dl/`. Antes de abrir o WebSocket Deriv, `bootstrap_and_validate_models` baixa `{symbol}.pth` e `latest_ts.pt`, executa forward pass de sanidade em TorchScript (`torchscript_sanity.verify_torchscript_artifact`) e falha rapido se o artefato estiver corrompido. Ver [`infra-docker.md`](infra-docker.md).

Redis local usa AOF `appendfsync everysec` (`infra/docker/redis.conf`). `make docker-up` aplica `host-prereq.sh` (`vm.overcommit_memory=1` no WSL).

---

### 2.1 Bootstrap

1. `app/run.py` carrega `config/settings.json` e PAT do `.env` (`AETHER_DERIV_PAT` + `AETHER_DERIV_APP_ID`).
2. `validate_infra_services` (quando `infra.enabled`) e `bootstrap_and_validate_models` (checkpoint + TorchScript + sanity).
3. `restore_orchestrator_state` e `AuthManager` abrem sessao REST/WebSocket via OTP PAT.
4. `Orchestrator` instancia stream, risco, executor e persistência.
5. `StreamHandler.start_candle_stream` busca histórico OHLC e assina velas (`style: candles`) e ticks (`style: ticks`).

### 2.2 Buffer e microestrutura

- `buffer_limit` limita velas em memória por símbolo.
- `history_bars` / `training_history_bars` definem recorte para treino e predição.
- `TickBuffer` agrega por barra fechada: contagem de ticks, intervalo médio, velocidade, aceleração e desvio padrão de diffs consecutivas.

### 2.3 Assinatura de estado de dados

Para evitar inferências duplicadas na virada de vela, o orquestrador usa `get_data_state_signature()`: concatena epoch e OHLC do último candle fechado por símbolo. Se a assinatura for idêntica ao ciclo anterior, o motor aguarda sem reprocessar.

---

## 3. Deep Learning

### 3.1 Labels e features

**Rótulo** (`dl_labels.py`, modo `ma_trend`):

```
Y[i] = 1.0 se média móvel suavizada indica alta na barra i + horizon
horizon = label_horizon_bars (padrão 1)
```

**19 features** (`FEATURE_DIM` em `dl_feature_build.py`):

| Grupo | Dim | Conteúdo |
|-------|-----|----------|
| Microestrutura | 5 | diff consecutiva, velocidade, aceleração, std diffs, ticks/barra |
| Tradicionais | 9 | RSI, delta-RSI, BB %B e width, ATR norm, distância EMA20/50, log-return, ROC |
| Volatilidade | 3 | vol rolling, vol vs alvo do símbolo, z-score |
| Persistência | 2 | Hurst (R/S), variance ratio |

Normalização anti-leakage: `fit_norm_stats` somente no split de treino walk-forward.

### 3.2 Modelo

- Arquitetura: **`tcn`** (padrão), **`lstm`** ou **`gru`** via `deep_learning.arch`.
- Saída: probabilidade bruta de alta (`raw_prob`).
- Checkpoint v4 em `data/dl/{symbol}.pth` + TorchScript `{symbol}_ts.pt` (espelho MinIO `latest_ts.pt`).
- `dl_symbol_runtime.py` prefere modelo scripted; fallback eager quando `_ts.pt` ausente.

### 3.3 Treino walk-forward

- Splits temporais com embargo (`dl_splits.py`).
- Early stopping pela perda de validação.
- Retreino: bootstrap de sessão, nova vela, rolling, forçado após loss.
- Treino deferido (`dl_deferred_train.py`): thread em background serializada.
- Deploy gate opcional (`dl_deploy_eval.py`): `deploy_ok=false` bloqueia execução.
- Gate de treinamento: símbolo sem treino da sessão recebe `gate_reason: training`.

### 3.4 Predição

`predict_symbol_decision` (`dl_predict.py`):

- Sempre `execute=True` quando a predição técnica é bem-sucedida.
- Calcula indicadores, trend (`dl_trend.py`) e enriquece métricas para o resolver.
- `gate_reason=None` após predição OK; bloqueio só em exceção (`predict_error`).
- Thresholds `confidence_call/put` (0.53/0.47) sao bases; com `dynamic_threshold.enabled`, flutuam por `bb_width`, `atr_norm` e regime de volatilidade.
- Grava `calibrated_prob`, `calibrated_edge` e thresholds dinamicos em metrics para resolver e quality gate.

`dl_gating.py` mantem utilitarios: `resolve_edge`, `resolve_calibrated_edge`, `direction_from_raw_prob`, `resolve_confidence_thresholds`.

`dl_calibration_fit.py` ajusta Platt logistico, isotonic (PAV) e temperatura no holdout (`deep_learning.calibration.method: auto`).

---

## 4. Direção e qualidade

### 4.1 Motor de direção (`execution_direction_resolver.py`)

Substitui travas binárias por scoring composto CALL vs PUT:

| Sinal | Comportamento |
|-------|---------------|
| `calibrated_prob` | Peso principal via `dl_raw_weight` (fallback: `raw_prob`) |
| `dynamic_call/put_threshold` | Pivot dinâmico para inferência DL |
| `val_accuracy` | Bias lateral |
| `trend_direction` + votos | `trend_bias` |
| RSI/Keltner extremos | `exhaustion_flip` (mean-reversion) |
| RSI+CMO+Keltner alinhados | `exhaustion_hard_gate` — atenua peso DL em 80% |
| Entropia de probabilidade | `execution_entropy_adaptive` — comprime `w_eff` em regime incerto |
| Hurst, ADX, vol_ratio, CMO | `indicator_regime` / `mean_reversion` |
| `val_accuracy` baixa | `low_val_flip` |

Retorna `(CALL|PUT, metrics)` com `direction_inverted`, `direction_margin`, `direction_hints`. Retorna `None` apenas em bloqueio técnico ou sem probabilidade calibrada/bruta.

Pesos configuráveis em `orchestrator.execution.direction_scoring`.

`execution_volatility_threshold.py` calcula `volatility_regime` e ajusta thresholds/edge por símbolo.

`execution_volatility_bb.py` aplica squeeze exponencial no edge minimo quando BB esta comprimido (`squeeze_edge_exponential_k`). Com `vol_ratio` abaixo de `vol_compression_threshold` (padrao 0.50), `vol_compression_hyperbolic_edge` eleva o edge por termos parabolico e hiperbolico (cap 0.12) para forcar SKIP em compressao extrema de volatilidade.

#### Camadas de exaustao

```mermaid
flowchart TD
  DL[_dl_call_put_scores] --> ENT[entropia adaptativa w_eff]
  ENT --> HG{hard gate RSI+CMO+Keltner?}
  HG -->|sim e ADX <= 0.40| ATT[dl_raw_weight x 0.20]
  HG -->|ADX > 0.40| BYP[sem atenuacao]
  HG -->|nao| RESOLVE[finalize_direction_metrics]
  ATT --> RESOLVE
  BYP --> RESOLVE
  RESOLVE --> SOFT[exhaustion_conflict_penalty]
  SOFT --> RANK[market_decision_score]
  SOFT --> QG[quality gate SKIP se penalty >= 0.12]
  ATT --> HPEN[exhaustion_penalty elevada SKIP hard gate]
```

| Camada | Modulo | Comportamento |
|--------|--------|---------------|
| Soft | `execution_exhaustion_conflict.py` | Penalidade leve pos-resolucao quando RSI e CMO conflitam com o lado DL |
| Hard | `execution_exhaustion_hard_gate.py` | Tripla RSI+CMO+Keltner `%B`; retencao 20% do peso DL; isencao com ADX > 0.40 |
| Flip | `_exhaustion_bias` no resolver | Mean-reversion com thresholds de `exhaustion_gate` (nao hardcoded) |

Config em `orchestrator.execution.exhaustion_gate`: `hard_gate_enabled`, `rsi_overbought` (0.73), `cmo_bull` (0.48), `keltner_overbought` (1.15), `dl_weight_retention` (0.20), `adx_super_trend_min` (0.40).

### 4.2 Gate de qualidade (`execution_quality_gate.py`)

Aplicado **após** resolução direcional em `_gather_cluster_candidates` e na validação final de `collect_cluster_orders`:

| Piso | Valor padrão | Efeito |
|------|--------------|--------|
| `mandatory_min_trade_score` | 0.68 | Score efetivo mínimo (modo normal) |
| `recovery_min_trade_score` | 0.64 | Piso em recovery (escala com perdas consecutivas) |
| `recovery_hurst_persistence_min` | 0.58 | Hurst minimo para persistencia em martingale N2+ |
| `recovery_hurst_log_scale` | 0.08 | Elevacao logaritmica do piso quando Hurst < 0.58 |
| `min_edge_execute` | 0.04 | Edge mínimo (usa `calibrated_edge`; respeita `dynamic_min_edge`) |
| `min_direction_margin` | 0.05 | Clareza CALL vs PUT no resolver |
| `inverted_min_score` | 0.74 | Score extra quando `direction_inverted=true` |
| `min_adx_normal` | 0.18 | ADX mínimo fora de recovery |

Ciclo sem candidato elegível → nenhuma ordem (qualidade > quantidade).

Em **recovery ativo** com `consecutive_losses >= 2`, o piso de `trade_score` e ajustado **por candidato** via `recovery_min_signal(..., hurst=indicators.hurst)` usando `recovery_hurst_adjusted_floor`. Apos SKIPs consecutivos por Hurst persistente, `recovery_skip_counter` no Redis reduz o limiar `recovery_hurst_persistence_min` linearmente (0.01/ciclo, piso 0.50). Se nenhum candidato do pool tiver Hurst acima do limiar efetivo, `execution_collect` retorna lista vazia (SKIP do ciclo inteiro).

### 4.3 Pool e seleção

- `execution_recovery_gate.cluster_entry_eligible` — bloqueio **somente técnico** + exige `raw_prob` ou direção.
- `execution_collect_helpers.recovery_hurst_blocks_collect` — filtro de persistencia Hurst antes da selecao.
- `execution_direction.build_execution_candidate` — delega ao resolver.
- `execution_symbols.select_best_execution_candidate` — ranking por `market_decision_score` (penaliza inversão e margem baixa).
- `execution_mandatory_pick` / `execution_direction_fallback` — fallbacks quando `mandatory_trade_each_cycle: true`.

### 4.4 Logs DL (`dl_cycle_brief.py`, `dl_cycle_log.py`)

Linha curta por ciclo:

```
DL | exec R_10:CALL c=0.86 | bias R_50:PUT c=0.62→CALL(trend) | 1 bloq
DL REC | exec R_25:PUT c=0.68 +2
```

Resumo detalhado em DEBUG; deduplicação via `build_dl_cycle_brief_key`.

---

## 5. Execução

### 5.1 Fases

- **FASE TREINO** — `_training_phase_gate` suspende ordens até `session_trained` em todos os símbolos.
- **FASE OPERACAO** — `collect_cluster_orders` seleciona melhor candidato ou retorna lista vazia.

### 5.2 ExecutionManager

- Monta ordens com stake de `RiskManager.calculate_stake`.
- Settlement assíncrono; reentrada via `post_settlement_cycle`.
- Contratos via `TradeHandler.buy_with_parameters`: RISE_FALL, 180 s.

---

## 6. Gerenciamento de risco

| Mecanismo | Módulo / config |
|-----------|-----------------|
| Kelly fracionário | `stake_sizing.py`, `kelly.fraction` |
| Penalidade consenso Kelly | `consensus_stake_penalty.py` — reduz `f*` quando `order_direction` diverge da maioria de votos tecnicos (`call_votes`/`put_votes`), ponderando `di_diff` e `cmo` opostos |
| Stop win diário | `stop_win_target.py` |
| Martingale recovery | `martingale_gate.py`, `martingale_conviction.py` |
| Trava Hurst N2+ | `recovery_hurst_gate.py` — piso logaritmico e filtro de pool |
| Decaimento Hurst em inanição | `recovery_hurst_decay.py` — `recovery_skip_counter` no Redis reduz `recovery_hurst_persistence_min` 0.01/ciclo ate 0.50 |
| Cooldown entrada | `risk_cooldown.py` |
| Cooldown por loss no símbolo | `symbol_loss_cooldown.py` |

Persistência: `data/session_state.json` (métricas diárias), `data/state.json` (contratos). Stop win no boot exige `total_trades_today > 0` para evitar falso positivo.

---

## 7. Orquestrador

`Orchestrator` (`orchestrator/__init__.py`):

1. `setup_trading_session` — autenticação Deriv e WebSocket.
2. A cada vela do âncora ou `cycle_interval_seconds`:
   - `tick_bars_since_train`
   - `collect_deep_learning_decisions`
   - `executor.execute_cluster`
3. Reconciliação periódica de contratos abertos.
4. Após `RECOV: WebSocket restaurado`, `reconcile_after_ws_recovery` audita portfolio/profit_table, persiste estado e bloqueia ciclo DL enquanto `_reconciliation_pending`.
5. Após liquidação: `post_settlement_cycle`.

Banner: `decision_mode_banner.emit_decision_engine_banner`.

---

## 8. Configuração crítica

| Bloco | Chaves relevantes |
|-------|-------------------|
| `data_handler` | `granularity`, `history_bars`, `fetch_count`, `buffer_limit` |
| `deep_learning` | `arch`, `lookback`, `confidence_*`, `min_val_accuracy`, `min_edge_execute`, `deploy_gate` |
| `orchestrator.execution` | `direction_scoring`, `quality_gate`, `exhaustion_gate`, `dynamic_threshold`, `mandatory_trade_each_cycle`, settlement |
| `risk_management.kelly` | `fraction`, `consensus_penalty_*`, martingale |
| `risk_management.params` | `duration` (180), stakes |
| `symbols` / `anchor` | Universo Range Break |
| `trading` | `mode` (`demo` / `live`) |

---

## 9. Camadas de software

| Camada | Módulos principais |
|--------|-------------------|
| Application / DL | `decision_bridge`, `dl_predict`, `dl_gating`, `dl_trend`, `dl_cycle_*`, `model` |
| Application / Direção | `execution_direction_resolver`, `execution_entropy_adaptive`, `execution_exhaustion_conflict`, `execution_exhaustion_hard_gate`, `execution_direction`, `execution_quality_gate`, `execution_volatility_bb` |
| Application / Execução | `execution_collect`, `execution_collect_helpers`, `execution_market_rank`, `execution_symbols`, `execution_mandatory_pick`, `execution_direction_fallback` |
| Application / Orchestrator | `Orchestrator`, `execution_manager`, `execution_recovery_gate`, `settlement_*`, `post_settlement_cycle` |
| Domain | `trade`, `market_data`, `probability_entropy`, `risk_manager`, `consensus_stake_penalty`, `recovery_hurst_gate`, `martingale_*`, `stake_sizing` |
| Infrastructure | `websocket_manager`, `stream_handler`, `tick_buffer`, `trade_handler`, `minio_model_store`, `torchscript_sanity`, `redis_state_pipeline`, `persistence_manager` |
| Presentation | `logger`, `log_dedupe` |

---

## 10. Observabilidade

| Ferramenta | Caminho |
|------------|---------|
| Log ao vivo | `logs/engine.log` |
| Monitor Rich | `app/scripts/monitor/live_monitor.py` |
| CI local | `app/scripts/operations/clean_workspace.py` |
| PAT Deriv | `app/scripts/operations/deriv_pat_connect.py` |
| Reset demo | `app/scripts/operations/reset_demo_balance.py` |

---

## 11. Garantia de qualidade

- Cobertura **100%** em `app/src` (pytest + coverage).
- Pre-commit: Ruff, Interrogate, Vulture, pylint duplicate-code, máximo **300 linhas** por arquivo em `app/src`.

---

## 12. Referências

- [medallion.md](medallion.md) — princípios quant e perfil de qualidade
- [README.md](../README.md) — execução e pré-requisitos
- [deriv-api.md](deriv-api.md) — API Deriv
- [infra-docker.md](infra-docker.md) — stack Docker, Redis AOF, host WSL
- [CHANGELOG.md](CHANGELOG.md) — histórico de releases
