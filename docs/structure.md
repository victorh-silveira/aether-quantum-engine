# Estrutura do repositório

Layout de software (sem infraestrutura de nuvem neste repo).

```
aether-quantum-engine/
├── app/
│   ├── src/
│   │   ├── application/services/
│   │   │   ├── deep_learning/     # TCN/LSTM/GRU, labels, Hurst, gating, deploy, decision_bridge
│   │   │   ├── orchestrator/      # Ciclo, execução, settlement, engine_session, engine_mode
│   │   │   ├── execution_*.py     # Direção, ranking de mercado, seleção e recovery
│   │   │   ├── log_dedupe.py      # Deduplicação de logs repetidos
│   │   │   └── auth_manager.py
│   │   ├── domain/                # Modelos, risk_manager, martingale, stake
│   │   ├── infrastructure/        # WebSocket, stream, tick_buffer, trade, persistência
│   │   └── presentation/          # Logger terminal
│   ├── tests/unit/                # Pytest (cobertura 100% em src)
│   ├── scripts/
│   │   ├── batch/                 # launch-all-demo, launch-train, _run_engine
│   │   ├── monitor/               # live_monitor
│   │   ├── operations/            # clean_workspace, deriv_pat_connect, reset_demo_balance
│   │   └── wsl/                   # setup.sh (WSL)
│   ├── aether_paths.py
│   ├── run.py                     # Entrada de execução (modo execute)
│   └── train.py                   # Entrada de treino DL (modo train)
├── config/settings.json           # Configuração versionada
├── data/                          # state.json, dl/, deriv/ (runtime)
├── logs/                          # engine.log, monitor.log
├── docs/
├── linters/                       # pre-commit, commitlint, release
├── .github/workflows/             # CI
├── run.py                         # Atalho para app/run.py
└── train.py                       # Atalho para app/train.py
```

## Camadas em `app/src`

| Pasta | Responsabilidade |
|-------|------------------|
| `application/services/deep_learning` | `dl_labels`, `dl_hurst`, `dl_feature_build` (19D), TCN/LSTM/GRU, gating por threshold 0.75/0.25, deploy gate, TorchScript |
| `application/services/orchestrator` | `Orchestrator`, `ExecutionManager`, `engine_session`, `engine_mode`, settlement, `post_settlement_cycle` |
| `application/services` | `execution_direction`, `execution_market_rank`, `execution_mandatory_pick`, `execution_symbols`, `execution_symbols_recovery`, `log_dedupe`, `auth_manager` |
| `domain` | `Candle`, `Trade`, `RiskManager`, Kelly, martingale, cooldowns, `stake_sizing` |
| `infrastructure` | `WebSocketManager`, `StreamHandler`, `TickBuffer`, `TradeHandler`, `PersistenceManager` |
| `presentation/terminal` | `setup_logger`, `BlankLineSquasher`, formatação de logs |

Decisão exclusivamente Deep Learning quando `deep_learning.enabled` é verdadeiro. Treino e execução são processos separados (`train.py` / `run.py`).

## Dados e artefatos

| Caminho | Uso |
|---------|-----|
| `data/state.json` | Estado geral de contratos e banca |
| `data/session_state.json` | Métricas diárias da sessão de trading e limites de Stop Win |
| `data/dl/{symbol}.pth` | Checkpoints PyTorch + calibrador + métricas |
| `data/dl/{symbol}_ts.pt` | TorchScript trace para inferência rápida |
| `data/deriv/pat_bindings.json` | Cache PAT → App ID |
| `logs/engine.log` | Auditoria operacional |

Caminhos resolvidos por `aether_paths.repo_path()` a partir da raiz do repositório.

## Comandos úteis (WSL)

Primeira vez no WSL: `make setup-wsl` (Git, Conda no `~/.bashrc`, hooks).

```bash
make install
make test
make lint
make train
make run
make clean
```

Pre-commit: `make pre-commit` instala hooks; `git commit` dispara lint, testes e segurança.
