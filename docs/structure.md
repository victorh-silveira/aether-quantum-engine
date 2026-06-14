# Estrutura do repositório

Layout de software (sem infraestrutura de nuvem neste repo).

```
aether-quantum-engine/
├── app/
│   ├── src/
│   │   ├── application/services/
│   │   │   ├── deep_learning/     # TCN/LSTM/GRU, labels, Hurst, gating, deploy, decision_bridge
│   │   │   ├── orchestrator/      # Ciclo, fases treino/operação, execução, settlement
│   │   │   ├── execution_*.py     # Direção, ranking de mercado, seleção e recovery
│   │   │   ├── log_dedupe.py      # Deduplicação de logs repetidos
│   │   │   └── auth_manager.py
│   │   ├── domain/                # Modelos, risk_manager, martingale, stake
│   │   ├── infrastructure/        # WebSocket, stream, tick_buffer, trade, persistência
│   │   └── presentation/          # Logger terminal
│   ├── tests/unit/                # Pytest (cobertura 100% em src)
│   ├── scripts/
│   │   ├── monitor/               # live_monitor
│   │   └── operations/            # clean_workspace (lint/test CI local)
│   ├── data/dl/                   # Checkpoints .pth e TorchScript _ts.pt por símbolo
│   ├── aether_paths.py
│   └── run.py
├── config/settings.json           # Configuração versionada
├── data/                          # state.json (runtime)
├── logs/                          # engine.log
├── docs/
├── linters/                       # pre-commit, commitlint, release
├── .github/workflows/             # CI
├── run.py                         # Atalho para app/run.py
└── Makefile
```

## Camadas em `app/src`

| Pasta | Responsabilidade |
|-------|------------------|
| `application/services/deep_learning` | `dl_labels`, `dl_hurst`, `dl_feature_build` (18D), TCN/LSTM/GRU, treino walk-forward deferido, gating por threshold 0.75/0.25, deploy gate, TorchScript |
| `application/services/orchestrator` | `Orchestrator`, `ExecutionManager` (fases treino/operação), `execution_collect`, settlement, `post_settlement_cycle` |
| `application/services` | `execution_direction`, `execution_market_rank`, `execution_mandatory_pick`, `execution_symbols`, `execution_symbols_recovery`, `log_dedupe`, `auth_manager` |
| `domain` | `Candle`, `Trade`, `RiskManager`, Kelly, martingale, cooldowns, `stake_sizing` |
| `infrastructure` | `WebSocketManager`, `StreamHandler`, `TickBuffer`, `TradeHandler`, `PersistenceManager` |
| `presentation/terminal` | `setup_logger`, `BlankLineSquasher`, formatação de logs |

Decisão exclusivamente Deep Learning quando `deep_learning.enabled` é verdadeiro. Estratégia TREND_FIBO e módulos legados (consensus, regime, pair features, binary_signal) foram removidos.

## Módulos DL principais

| Arquivo | Função |
|---------|--------|
| `dl_labels.py` | Rótulos binários alinhados ao contrato (horizon = 1 barra) |
| `dl_hurst.py` | Hurst exponent e variance ratio |
| `dl_feature_build.py` | 18 features (micro + tradicionais + vol + persistência) |
| `dl_tcn.py` / `dl_lstm.py` | Arquiteturas de rede |
| `model.py` | Factory TCN/LSTM/GRU, checkpoint v3, TorchScript |
| `dl_gating.py` / `dl_predict.py` | Threshold 0.75/0.25 e predição |
| `tick_buffer.py` | Agregação de microestrutura por barra |

## Dados e artefatos

| Caminho | Uso |
|---------|-----|
| `data/state.json` | Estado de contratos e banca (via `repo_path`) |
| `data/dl/{symbol}.pth` | Checkpoints PyTorch + calibrador + métricas |
| `data/dl/{symbol}_ts.pt` | TorchScript trace para inferência rápida |
| `logs/engine.log` | Auditoria operacional |

Caminhos resolvidos por `aether_paths.repo_path()` e `APP_ROOT`.

## Comandos úteis (WSL)

Primeira vez no WSL: `make setup-wsl` (Git, Conda no `~/.bashrc`, hooks).

```bash
make install
make test
make lint
make run
make clean
```

Pre-commit: `make pre-commit` instala hooks; `git commit` dispara lint, testes e segurança.
