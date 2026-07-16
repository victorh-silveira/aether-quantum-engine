# Arquitetura — Aether Quantum Engine

Motor assíncrono para trading na Deriv com decisão por **Deep Learning** (TCN, LSTM ou GRU) nos índices **Drift** (`RDBEAR`, `RDBULL`). Metodologia quantitativa: [`medallion.md`](medallion.md). Inventário de módulos: [`structure.md`](structure.md). Infra Docker: [`infra-docker.md`](infra-docker.md).

---

## 1. Visão geral

| Aspecto | Valor atual (`config/settings.json`) |
|---------|--------------------------------------|
| Símbolos | `RDBEAR`, `RDBULL` (âncora `RDBULL`) |
| Granularidade OHLC (DL) | **900 s / M15** (`data_handler.granularity`) |
| Relógio operacional | **60 s / M1** (`data_handler.micro_granularity`) |
| Histórico para treino | 15552 barras M15 (`training_history_bars`, ~162 dias) |
| Lookback | 48 barras M15 → tensor **`[1, 48, 34]`** (~12 h) |
| Features TCN | **34** (`FEATURE_DIM` em `dl_feature_build.py`) |
| Features meta GBDT | **43** (`META_FEATURE_DIM` = 34 + 4 micro-vol + 3 cross + 2 flow) |
| Contrato | `RISE_FALL`, duração **60 s** |
| Ciclo | 60 s (`cycle_interval_seconds`) |
| Execução | **Esteira mandatária** (`mandatory_trade_each_cycle: true`) |
| Fail-closed | `require_meta_for_execution: true`, `infra.triton.require_for_execution: true` |
| Label | `label_mode: ma_trend` (Triple Barrier disponível via config) |
| Meta sessão | Stop win **2,60%** (`compounding_rate_daily: 0.026`); stop loss desativado |

O mercado é tratado como série temporal ruidosa: a TCN estima `P(CALL)`; o meta-regressor LightGBM estima `predicted_payoff_edge`; o ranking usa `tcn × max(0.1, 1+z)`. Em modo mandatário, o motor exige mandatory pick quando há candidatos aprovados pelo quality gate dual (TCN + meta).

**Invariante temporal:** inferências seguem `signature_boundary_seconds` (fallback `cycle_interval_seconds`) via `get_data_state_signature()` — formato `m1b:{boundary};m1:...;m15:...`.

**Válvula de starvation:** após **6** ciclos consecutivos bloqueados pelo quality gate (`STARVATION_DECAY_THRESHOLD`), pisos de margem/edge/Z são atenuados (`execution_quality_gate_starvation.py`). Em skips extremos (≥30), a válvula GBDT mitiga veto tabular prolongado.

---

## 2. Layout e camadas DDD

```
aether-quantum-engine/
├── app/                 # Código de produção + testes + scripts
│   ├── run.py / train.py
│   ├── src/             # ~219 módulos Python (DDD)
│   ├── tests/           # ~268 arquivos test_*.py; cobertura 100% em src
│   └── scripts/         # operations, monitor, batch
├── config/settings.json # Runtime
├── data/                # state, session_state, dl/
├── docs/
├── infra/docker/        # Redis, Timescale, MinIO, Triton, meta-classifier
└── linters/             # Ruff, Interrogate, Vulture, ≤300 linhas/arquivo
```

```
presentation  →  application  →  domain
                    ↓
              infrastructure (ports/adapters)
```

| Camada | Pasta | Papel |
|--------|-------|-------|
| Application | `application/services/` | Orquestração, DL, direção, quality gates, meta |
| Domain | `domain/` | Risco Kelly/D'Alembert, `RiskPolicy`, modelos, math |
| Infrastructure | `infrastructure/` | Deriv WS/REST, Redis, Triton, MinIO, Timescale |
| Presentation | `presentation/` | Logger terminal |

Regra: **domain** não importa application nem infrastructure. Implementações concretas vêm de `infra_factory.create_infra_services`.

---

## 3. Pipeline de runtime

```mermaid
flowchart TD
  RUN[app/run.py] --> ORCH[Orchestrator]
  ORCH --> BOOT[ws_bootstrap + bootstrap_and_validate_models]
  BOOT --> LOOP[run_orchestrator_main_loop]
  LOOP --> TC[run_trading_cycle_if_ready]
  TC --> SIG[get_data_state_signature M1+M15]
  TC --> LOCK[orchestrator_atomic_state_context]
  LOCK --> DL[collect_deep_learning_decisions]
  DL --> TRI[TritonGrpcClient / force_local no deploy gate]
  DL --> BUNDLE[prepare_meta_classifier_cross_symbol_bundle]
  BUNDLE --> META[prefetch_meta_payoff_for_decisions]
  META --> RES[resolve_execution_direction]
  RES --> ATL[AntiTrendLock]
  ATL --> QG[quality_conviction_suspends_cluster]
  QG -->|skip| STV[record_quality_guard_cycle_skip]
  QG -->|ok| COL[collect_cluster_orders / execute_cluster]
  COL --> RM[RiskManager.calculate_stake]
  RM --> TH[TradeHandler.buy_with_parameters]
  TH --> SET[process_contract_settlement]
  SET --> RSC[risk_recovery_state + save_full_state]
  SET --> PSC[post_settlement_cycle / stop_win]
```

### 3.1 Bootstrap

1. `app/run.py` carrega `config/settings.json` + PAT (`.env`: `AETHER_DERIV_PAT`, `AETHER_DERIV_APP_ID`).
2. `Orchestrator.__init__`: `create_infra_services`, `AuthManager`, `WebSocketManager`, `StreamHandler`, `TradeHandler`, `RiskManager`, `StateManager`, `ExecutionManager`.
3. `validate_engine_risk_config` / `RiskPolicy` no boot (`risk_policy.py`).
4. `validate_infra_services` (fail-fast se `infra.enabled`) → `bootstrap_and_validate_models`:
   - MinIO → `{symbol}.pth` + `latest_ts.pt`
   - Sanity TorchScript multi-probe (`torchscript_sanity_probes`)
   - Sync Triton (`triton_model_sync`) + schema + stress infer
5. Auth OTP → `bootstrap_active_session_targets` (meta 2,60%).
6. Streams M15+M1+ticks → watchdog → settlement worker → loop principal.

### 3.2 Ciclo de trading

`trading_cycle_entry.run_trading_cycle_if_ready`:

1. `process_redis_settlement_queue`
2. Guards (`trading_cycle_entry_guards`): stop-win, assinatura, cadência, contratos ativos
3. `prepare_quality_skipped_cycles_counter` (Redis `state:risk:skipped_cycles_counter`)
4. Sob `orchestrator_atomic_state_context`:
   - `collect_deep_learning_decisions`
   - `quality_conviction_suspends_cluster` → skip + incremento starvation **ou**
   - `executor.execute_cluster(decisions)` → reset starvation

**Lock atômico** (`StateManager._state_lock`): inferência + boleta **não** correm em paralelo com liquidação/persistência de risco.

### 3.3 Assinatura M1+M15

| Componente | Função |
|------------|--------|
| `m1_boundary_epoch()` | Epoch alinhado ao minuto corrente |
| Micro | `m1:{sym}@{epoch}` |
| Macro | `m15:{sym}@{epoch}` |
| Formato | `m1b:{boundary};m1:...;m15:...` |

Cache inválido quando a assinatura muda; sem assinatura nova, o ciclo aguarda sem re-inferir.

---

## 4. Deep Learning

### 4.1 Features (34D)

| Grupo | Dim | Conteúdo |
|-------|-----|----------|
| Microestrutura | 5 | ticks/barra, intervalo, velocidade, aceleração, std diffs |
| Tradicionais | 22 | RSI, BB, ATR, EMAs, MACD, estocástico, CCI, ADX, Williams, CMO, Keltner, etc. |
| Volatilidade | 5 | vol rolling, vol vs alvo, z-score, implied vol, vol_ratio |
| Persistência | 2 | Hurst, variance ratio |

Módulos: `dl_feature_build.py` (séries), `dl_feature_matrix.py` (linhas/matrizes/tensores), `dl_feature_indicators*.py`, facade `dl_features.py`.

Normalização anti-leakage: `fit_norm_stats` **somente** no split de treino (`dl_splits` purged/embargo).

### 4.2 Modelo e labels

| Peça | Path |
|------|------|
| TCN | `dl_tcn.TemporalDirectionClassifier` (+ `regression_head` multi-task) |
| LSTM/GRU | `dl_lstm.py`; `deep_learning.arch` |
| Perda | Focal assimétrica alta vol (`dl_training_epochs._masked_loss`) |
| Labels | `dl_labels.LabelSpec` — `spot_forward` / `ma_trend` / `triple_barrier` |
| Checkpoint | `data/dl/{symbol}.pth` + TorchScript / MinIO |
| Deploy gate | `dl_deploy_eval` com **`force_local=True`** (avalia modelo em memória, sem Triton) |

Config atual: `arch: tcn`, `lookback: 48`, `label_mode: ma_trend`, thresholds base `0.53` / `0.47`.

### 4.3 Treino de sessão

- Walk-forward: `dl_symbol_train` → `train_model_walkforward`
- Bootstrap sequencial: `dl_bootstrap_train` / `run_dl_training_session` via `asyncio.to_thread`
- Retreino deferido: `dl_deferred_train` (thread + semáforo)
- CUDA: init no main thread (`dl_device`); uploads MinIO via `loop.call_soon_threadsafe`
- Gate: `training` até `session_trained` (`dl_training_gate`)

### 4.4 Predição

`predict_symbol_decision_async` (`dl_predict_async.py`):

- Triton quando habilitado e **não** `force_local`
- Timeout `infra.triton.infer_timeout_seconds` (**0,85 s**); com `require_for_execution: true` → fail-closed sem fallback local em produção
- Path sync (`dl_predict.py`) usado pelo mini-deploy de treino: `force_local=True` evita gRPC em thread de treino (bug `Future attached to a different loop`)
- Cliente gRPC loop-aware: `triton_grpc_client.get_triton_grpc_client` recria canal se o event loop mudou
- Cache por fingerprint do tensor (`dl_predict_cache`)

---

## 5. Meta-classificador (LightGBM)

### 5.1 Serviço

| Item | Valor |
|------|-------|
| Container | `aether-meta-classifier` (FastAPI), host **8005→8000** |
| Endpoint | `POST /v2/predict_meta` |
| Cliente | `MetaClassifierClient` + `meta_classifier_pool` (rebind por loop) |
| Timeout | 1,0 s; fallback `predicted_payoff_edge=0.0` |
| Artefatos | `infra/docker/meta-models/*.pkl` |

### 5.2 Vetor 43D

```
META_FEATURE_DIM = 34 (TCN) + 4 (micro-vol zscores) + 3 (cross-symbol) + 2 (flow) = 43
```

| Bloco | Features |
|-------|----------|
| Micro-vol | `micro_bid_ask_spread_momentum[_zscore]`, `volatility_shadow_ratio[_zscore]` |
| Cross | `cross_symbol_prob_delta`, `cross_symbol_vol_ratio_diff`, `cross_symbol_rsi_spread` |
| Flow | `micro_tick_acceleration`, `keltner_deviation_ratio` |

Montagem: `dl_predict_telemetry.prepare_meta_classifier_cross_symbol_bundle` → `extract_meta_feature_vector` → `prefetch_meta_payoff_for_decisions`.

**Nota:** o container Docker legado pode declarar `FEATURE_DIM=39` (sem o bloco micro-vol de 4). O **app** é a fonte de verdade (**43D**). Treino offline e artefato `.pkl` devem alinhar com `META_FEATURE_DIM`.

### 5.3 Stacking runtime

1. Bundle cross-symbol + telemetria micro M1
2. Prefetch HTTP → `predicted_payoff_edge`
3. `attach_payoff_edge_zscore_metrics` (janela adaptativa 15–45)
4. `apply_meta_regression_edge`: edge > 0 mantém score TCN; edge < −0,15 + squeeze → `trade_score=0.52` (`[D-SQUEEZE]`)
5. Ranking: `market_decision_score = tcn × max(0.1, 1+z)`
6. Veto Z negativo: `meta_payoff_veto_gate` (waiver em recovery crítico)

### 5.4 Treino offline e anti-leakage

Scripts: `train_meta_vector.py`, `train_meta_data.py`, `train_meta_classifier.py`, `train_meta_optuna.py`.

- Alvo: `Y ≈ PnL / Stake`
- Optuna maximiza Information Ratio; constraint Z-Score OOS
- Anti-leakage: proxy de probabilidade via **retorno passado** (não label forward do horizonte de execução)
- Alinhamento temporal por epoch em `train_meta_vector`

---

## 6. Direção e quality gates

### 6.1 Motor de direção

`execution_direction_resolver.resolve_execution_direction`:

| Etapa | Comportamento |
|-------|---------------|
| `infer_dl_direction` | TCN: `P(CALL) > pivot` → CALL, senão PUT |
| Meta edge | Refina score / D-SQUEEZE |
| AntiTrendLock | Após 2 losses na mesma direção: FLIP cross-symbol ou FREEZE (`evaluate_anti_trend_lock`) |
| Bloqueio absoluto | `deploy_ok=false`, `gate_reason ∈ {data, predict_error, training}` |

### 6.2 Quality gate dual

| Portão | Módulo | Critério |
|--------|--------|----------|
| TCN + meta soft | `execution_quality_gate` | Margem direcional + edge (quando meta aplicado) |
| Meta Z-Score | `execution_quality_gate_meta` | Z vs buffer; recovery: Z ≥ −0,20 não bloqueia mandatory |
| Cluster | `execution_quality_gate_cluster` | `quality_conviction_suspends_cluster` |
| Starvation | `execution_quality_gate_starvation` | Limiar **6** ciclos; fator `max(0.20, 1 − (n−5)×0.10)` |

Em modo mandatário, o quality guard emite telemetria `QUALITY_GUARD` / `EXECUTION_FLOW` e delega ao mandatory pick em vez de congelar a esteira.

### 6.3 Ranking e seleção

- `execution_market_rank.market_decision_score`
- Redirect inter-símbolo: âncora Z < −0,50 → par Z > +0,50 (`try_inter_symbol_zscore_redirect`)
- `execution_collect` / `execution_mandatory_pick` / `execution_symbols`
- Cointegração Drift sob drawdown: `apply_cointegration_redirect`

---

## 7. Execução e settlement

### 7.1 Fases

- **FASE TREINO** — suspende ordens até `session_trained` em todos os símbolos
- **FASE OPERACAO** — esteira mandatária contínua

### 7.2 ExecutionManager

- Stake via `RiskManager.calculate_stake` → `risk_stake_calc.calculate_stake_for_manager`
- Lotes fracionados se stake > `max_single_stake_limit` (`execution_fractional_lots`)
- `TradeHandler.buy_with_parameters`: RISE_FALL 60 s
- Reconciliação de stake downgrade Deriv (`executed_stake_reconciliation`)

### 7.3 Settlement

```mermaid
flowchart LR
  POC[proposal] --> IQ[asyncio.Queue]
  IQ --> WK[settlement worker]
  WK --> PCS[process_contract_settlement]
  PCS -->|ws offline| ZSET[Redis ZSET settlement:queue:priority]
  PCS -->|ws online| RSC[risk_recovery_state]
  RSC --> SAVE[save_full_state MULTI/EXEC]
```

Pós-liquidação (`post_settlement_cycle`): stop-win fast-path limpa Redis, cancela fila e `graceful_shutdown(fast_path=True)`; senão retry com teto e recovery transparente de deadlock.

Portões neutralizados em modo mandatário (não bloqueiam ciclo): cooldown pós-LOSS, blackout API, Hurst recovery collect, freeze yield.

---

## 8. Gerenciamento de risco

| Mecanismo | Módulo |
|-----------|--------|
| Kelly fracionário | `kelly_base_fraction`, `stake_sizing`; `kelly.fraction: 0.005` |
| Teto de stake | `max_stake_pct` / `max_bankroll_stake_fraction: 0.035`; `max_safe_stake_cap` pós-turbo |
| Consensus entropy | `consensus_stake_penalty.consensus_kelly_retention` |
| Soft recovery | `apply_soft_recovery_stake` + progressão adaptativa por payout (não `2^n` puro sem teto) |
| Cap de segurança | `max_safe_stake_cap(bankroll, consecutive_losses_linear=…)` após turbo |
| Recovery persistente | `pending_loss` + `consecutive_losses_linear`; reset só quando passivo zera |
| Retração WIN parcial | `linear = max(1, n−1)` |
| Stop win sessão | `StopWinManager` + `compounding_rate_daily: 0.026` |
| Stop loss | Desativado |
| Cover pending | Fracionado por `amort_cycles` |
| Policy boot | `RiskPolicy` / `validate_engine_risk_config` |
| AntiTrendLock | `evaluate_anti_trend_lock` → KEEP / FLIP / FREEZE |

Facade: `domain/risk/risk_manager.RiskManager.calculate_stake`.

**Sessão:** meta = `session_start_balance × 0.026`. Log: `SESSAO INICIADA | Alvo de 2.60%: $… | Stop Loss: DESATIVADO`. Reiniciar o processo inicia sessão nova.

---

## 9. Infraestrutura

| Serviço | Porta | Uso |
|---------|-------|-----|
| Redis | 6379 | Estado atômico, settlement ZSET, starvation counter |
| TimescaleDB | 5432 | Ticks/OHLC; correlação |
| MinIO | 9000/9001 | Checkpoints TCN/TorchScript |
| Triton | 8000/8001 | Inferência GPU gRPC |
| Meta-classifier | 8005 | LightGBM HTTP |
| Deriv | WS/REST | Auth PAT, streams, propostas |

Config `infra.*`: `enabled`, `fail_fast`, `redis`, `timescale`, `minio`, `triton` (`infer_timeout_seconds: 0.85`, `require_for_execution`), `meta_classifier`.

Persistência: `redis_state_pipeline.write_state_bundle` (MULTI/EXEC) — snapshot, risk hash, pending_loss, session keys, skip counters, market_sig, settlement queue.

Watchdog: `AetherWatchdog` reconecta stream se ticks estagnarem (`watchdog_stale_tick_seconds`, padrão 30 s).

---

## 10. Configuração (`config/settings.json`)

### `data_handler`
`granularity` (900), `micro_granularity` (60), `history_bars` / `training_history_bars` (15552), `fetch_count`, `buffer_limit`, rate-limits de histórico.

### `deep_learning`
`arch`, `lookback`, `train_symbols`, `confidence_*`, `calibration.*`, `deploy_gate.*`, `label_mode` + `label_*`, `tcn.channels`, `training_*`, `model_path_template`, `min_edge_execute`.

### `orchestrator` / `orchestrator.execution`
`cycle_interval_seconds`, `watchdog_*`, `mandatory_trade_each_cycle`, `require_meta_for_execution`, `quality_gate.*`, `loss_protection.*`, `proposal_*`, `settlement_*`, `dynamic_threshold.*`.

### `risk_management`
`kelly.*` (fraction, caps, consensus, recovery Hurst), `dlambert.*`, `params.*` (duration 60, compounding, stake_min, payout_estimate).

### `infra`
Redis/Timescale/MinIO/Triton/meta_classifier URLs e timeouts.

---

## 11. Observabilidade e QA

Marcadores de log frequentes: `QUALITY_GUARD`, `EXECUTION_FLOW`, `REGIME_GUARD`, `D-SQUEEZE`, `TRITON_TIMEOUT_*`, `META_CLASSIFIER_FALLBACK`, `STOP_WIN`, `SESSAO INICIADA`, `DATA_SIG`.

Pre-commit (`clean_workspace.py`):

| Stage | Conteúdo |
|-------|----------|
| lint | Ruff, Interrogate 100%, Vulture, ≤300 linhas/arquivo |
| test | pytest + cobertura **100%** em `app/src` |
| security | Bandit + pip-audit |
| clean | caches locais |

---

## 12. Diagrama condensado do software

```mermaid
flowchart LR
  subgraph ingestao
    WS[WebSocketManager]
    SH[StreamHandler]
    TB[TickBuffer]
    WD[AetherWatchdog]
  end
  subgraph dl
    FEAT[dl_features 34D]
    TRITON[TritonGrpcClient]
    PRED[dl_predict]
    TELE[dl_predict_telemetry]
    META[meta 43D GBDT]
  end
  subgraph direcao
    RES[direction_resolver]
    ATL[AntiTrendLock]
    QG[quality_gate dual]
  end
  subgraph exec
    COL[execution_collect]
    EM[ExecutionManager]
    TH[TradeHandler]
  end
  subgraph pos
    ST[settlement + Redis queue]
    RM[RiskManager]
    LOCK[StateManager Lock]
  end
  WS --> SH --> TB
  WD -->|STALE_DATA| SH
  SH --> FEAT --> TRITON --> PRED --> TELE --> META --> RES --> ATL --> QG --> COL --> EM --> TH
  TH --> ST --> RM
  COL --> LOCK
  ST --> LOCK
```
