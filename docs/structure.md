# Estrutura do repositório

Layout de software com infraestrutura Docker local opcional (`infra/docker/`).

```
aether-quantum-engine/
├── infra/docker/                       # Redis (AOF), TimescaleDB, MinIO, Triton (GPU), meta-classifier
├── app/
│   ├── src/
│   │   ├── application/services/
│   │   │   ├── deep_learning/          # TCN/LSTM/GRU, labels, predict, dl_predict_build, Triton bridge
│   │   │   ├── orchestrator/           # Ciclo, execução, settlement, settlement_queue_ops, post_settlement_cycle, watchdog, graceful shutdown
│   │   │   ├── execution_*.py          # Resolver linear, meta stacking, cross-symbol, qualidade neutra, ranking
│   │   │   ├── log_dedupe.py
│   │   │   └── auth_manager.py
│   │   ├── domain/                     # Modelos, risk_manager, stop_win_target, risk_recovery_state, consensus_stake_penalty, dlambert_sizing
│   │   ├── infrastructure/
│   │   │   ├── inference/              # triton_grpc_client, meta_classifier_client, triton_model_sync
│   │   │   ├── state/                  # redis_state_pipeline, redis_state_store
│   │   │   ├── storage/                # minio_model_store, torchscript_sanity, torchscript_sanity_probes
│   │   │   └── ...                     # WebSocket, stream, stream_reconnect, tick_buffer, trade
│   │   └── presentation/               # Logger terminal
│   ├── tests/unit/                     # Pytest (cobertura 100% em src)
│   ├── scripts/
│   │   ├── batch/                      # launch-all-demo, launch-train, _run_engine
│   │   ├── monitor/                    # live_monitor
│   │   ├── operations/                 # clean_workspace, deriv_pat_connect, train_meta_classifier, train_meta_optuna, train_meta_vector, reset_demo_balance
│   │   └── wsl/                        # setup.sh (WSL)
│   ├── aether_paths.py
│   ├── run.py                          # Entrada de execução (modo execute)
│   └── train.py                        # Entrada de treino DL (modo train)
├── config/settings.json                # Configuração versionada
├── data/                               # state.json, dl/, deriv/ (runtime)
├── logs/                               # engine.log, monitor.log
├── docs/
├── linters/                            # pre-commit, commitlint, release
├── .github/workflows/                  # CI
├── run.py                              # Atalho para app/run.py
└── train.py                            # Atalho para app/train.py
```

## Camadas em `app/src`

| Pasta | Responsabilidade |
|-------|------------------|
| `application/services/deep_learning` | Features **34D**, TCN/LSTM/GRU, `dl_predict_build` (bundle cross-symbol), `dl_predict_async`, `dl_predict_triton`, `dl_trend`, deploy gate, TorchScript |
| `application/services/orchestrator` | `Orchestrator`, `ExecutionManager`, `session_target_bootstrap`, `execution_collect`, `execution_recovery_gate`, `watchdog_service`, `graceful_shutdown`, `settlement_*`, `settlement_queue_ops`, `post_settlement_cycle`, `orchestrator_run_loop` |
| `application/services` | `execution_direction_resolver` (TCN + edge contínuo D-SQUEEZE), `meta_payoff_regression`, `meta_classifier_stacking`, `meta_classifier_features`, `meta_classifier_cross_symbol`, `meta_classifier_flow_features`, `meta_direction_flip` (auditoria), `execution_direction_cross_corr` (telemetria), `execution_quality_gate` (neutro), `execution_market_rank`, `execution_symbols`, `execution_mandatory_pick`, `log_dedupe`, `auth_manager` |
| `domain` | `Candle`, `Trade`, `RiskManager`, `StopWinManager` (`stop_win_target.py`), `risk_recovery_state`, `consensus_stake_penalty`, `recovery_hurst_gate`, `probability_entropy`, Kelly, `dlambert_sizing` (Martingale Geométrico), `recovery_conviction`, cooldowns, `stake_sizing` |
| `infrastructure/inference` | `TritonGrpcClient` (gRPC aio persistente), `triton_inference_client`, `meta_classifier_client`, `triton_model_sync`, `triton_tensor_builder`, `triton_model_metadata` |
| `infrastructure/state` | `redis_state_pipeline` (MULTI/EXEC atômico), `redis_state_store`, ports `StateStore` |
| `infrastructure/storage` | `minio_model_store`, `torchscript_sanity`, `torchscript_sanity_probes` (probes estressados) |
| `infrastructure` (demais) | `WebSocketManager`, `StreamHandler`, `stream_reconnect`, `TickBuffer`, `TradeHandler`, Timescale, MinIO |
| `presentation/terminal` | `setup_logger`, `BlankLineSquasher`, formatação de logs |

Decisão exclusivamente Deep Learning quando `deep_learning.enabled` é verdadeiro. Treino e execução são processos separados (`train.py` / `run.py`).

## Módulos de execução (pipeline atual)

```mermaid
flowchart TD
  BR[decision_bridge] --> BUNDLE[dl_predict_build cross-symbol bundle]
  BUNDLE --> PRED[dl_predict_triton]
  PRED --> META[meta_classifier_client GBDT M1]
  META --> RES[execution_direction_resolver TCN + edge contínuo]
  RES --> QG[execution_quality_gate neutro]
  QG --> COL[execution_collect]
  COL --> RANK[execution_market_rank / execution_symbols]
  RANK --> EM[ExecutionManager]
```

| Arquivo | Papel |
|---------|-------|
| `dl_predict_build.py` | `prepare_meta_classifier_cross_symbol_bundle` — telemetria micro M1 paralela + spreads cross-symbol antes do prefetch HTTP |
| `triton_grpc_client.py` | Canal `grpc.aio.insecure_channel` persistente; timeout 2 s por inferência; `infer_symbols_concurrent` via `asyncio.gather`; fallback TorchScript em timeout |
| `meta_classifier_client.py` | Consulta assíncrona httpx ao container porta 8005; timeout 1,0 s; retorna `predicted_payoff_edge`; fallback neutro em falha |
| `meta_payoff_regression.py` | Edge `> 0` preserva score TCN; edge `< -0.15` em squeeze rebaixa para `0.52`; log `[D-SQUEEZE]` |
| `meta_classifier_features.py` | Extração tabular 39D (34 TCN + 3 cross-symbol + 2 fluxo micro); `meta_classifier_column_names()` |
| `meta_classifier_cross_symbol.py` | Triplet cross-symbol: `prob_delta` (abs), `vol_ratio_diff` e `rsi_spread` (spreads lineares assinados) |
| `train_meta_vector.py` | Montagem vetorizada do dataset 39D com alinhamento epoch; alvo `Y = PnL_Real / Stake` |
| `train_meta_optuna.py` | Optuna + `LGBMRegressor` huber; minimiza MAE; `n_jobs=2`; `LOKY_MAX_CPU_COUNT=4` no boot |
| `watchdog_service.py` | `AetherWatchdog` — detecta inanição de ticks (>30 s) e reconecta stream sem derrubar o motor |
| `stream_reconnect.py` | Reabertura controlada de WS + subscrições OHLC/tick após STALE_DATA |
| `graceful_shutdown.py` | Encerramento ordenado; `fast_path=True` cancela fila de settlement e aborta tasks pós-liquidação |
| `settlement_queue_ops.py` | `cancel_settlement_queue_fast` — cancela worker e drena fila sem handshake `task_done` |
| `post_settlement_cycle.py` | Breath pós-liquidação; curto-circuito stop win; teto 2× ciclo incompleto |
| `orchestrator_run_loop.py` | Loop principal; `_enforce_post_settlement_deadlock_exit` → `emergency_save_session_state` + `sys.exit(0)` |
| `execution_direction_resolver.py` | TCN define `dl_direction`; meta-regressor refina via `predicted_payoff_edge` e downgrade D-SQUEEZE |
| `dlambert_sizing.py` (domain) | Martingale `Effective_Base × 2^n` com ancoragem `max(override, U)` |
| `risk_stake_calc.py` (domain) | Bypass de consensus penalty quando `pending_total > 0` |
| `execution_quality_gate.py` | Validação neutra de predição ativa; skip de ciclo desativado estruturalmente |
| `execution_collect_helpers.py` | `recovery_hurst_blocks_collect`, mandatory fallback, logs EXEC_SEL |
| `execution_collect.py` | Coleta e seleção contínua mandatória sem filtros de barreira |
| `consensus_stake_penalty.py` (domain) | Penalidade Kelly base por divergência técnica; waiver absoluto em regime de recovery ativo |
| `risk_recovery_state.py` (domain) | Reset do expoente consecutivo condicionado à extinção real de `pending_loss` |
| `redis_state_pipeline.py` | Escrita atômica MULTI/EXEC do bundle de risco, snapshot e chaves de sessão no Redis |
| `session_target_bootstrap.py` | Bootstrap e clearing de metas compostas de 1% atreladas ao processo vivo |
| `stop_win_target.py` (domain) | Cálculo centralizado de `target_win` e gerenciamento de estados de sessão composta |

## Dados e artefatos

| Caminho | Uso |
|---------|-----|
| `data/state.json` | Estado geral de contratos e banca (legado) |
| `data/session_state.json` | Métricas da sessão ativa corrente (`session_start_balance`, `target_win`, P&L) |
| `data/dl/{symbol}.pth` | Checkpoints PyTorch + calibrador + métricas |
| `data/dl/{symbol}_ts.pt` | TorchScript trace para inferência Triton |
| `infra/docker/triton-models/{symbol}/` | Layout Triton (`model.pt`, `config.pbtxt`) |
| `infra/docker/meta-classifier/` | Container FastAPI Python 3.13-slim; `POST /v2/predict_meta`; healthcheck `urllib` em `/health` (porta host **8005**) |
| `infra/docker/meta-models/` | Artefatos LightGBM do meta-regressor (`.pkl`, `FEATURE_DIM=39`); `LGBMRegressor` huber treinado com `feature_name` explícito |
| `data/deriv/pat_bindings.json` | Cache PAT → App ID |
| `logs/engine.log` | Auditoria operacional |

Caminhos resolvidos por `aether_paths.repo_path()` a partir da raiz do repositório.

## Comandos úteis (WSL)

Primeira vez no WSL: `make setup-wsl` (Git, Conda no `~/.bashrc`, hooks).

```bash
make install
make test
make lint
make docker-up
make train
make run
make clean
```

Pre-commit: `make pre-commit` instala hooks; `git commit` dispara lint, testes e segurança.
