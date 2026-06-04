# Estrutura do repositório

Layout de software (sem infraestrutura de nuvem neste repo).

```
aether-quantum-engine/
├── app/
│   ├── src/
│   │   ├── application/services/
│   │   │   ├── deep_learning/     # TCN, treino, gating, deploy, decision_bridge
│   │   │   ├── orchestrator/      # Ciclo, execução, settlement
│   │   │   ├── execution_*.py     # Direção e seleção de símbolos
│   │   │   └── auth_manager.py
│   │   ├── domain/                # Modelos, risk_manager, martingale, stake
│   │   ├── infrastructure/        # WebSocket, stream, trade, persistência
│   │   └── presentation/          # Logger terminal
│   ├── tests/unit/                # Pytest (cobertura 100% em src)
│   ├── scripts/
│   │   ├── backtest/              # dl_walkforward, medallion (legado)
│   │   ├── monitor/               # live_monitor
│   │   └── operations/            # clean_workspace (lint/test CI local)
│   ├── data/dl/                   # Checkpoints .pth por símbolo
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
| `application/services/deep_learning` | Features, TCN, treino walk-forward, calibração, predição, deploy gate |
| `application/services/orchestrator` | `Orchestrator`, `ExecutionManager`, settlement, sessão de trading |
| `application/services` | `execution_direction`, `execution_symbols`, `auth_manager` |
| `domain` | `Candle`, `Trade`, `RiskManager`, Kelly, martingale, cooldowns |
| `infrastructure` | `WebSocketManager`, `StreamHandler`, `TradeHandler`, `PersistenceManager` |
| `presentation/terminal` | `setup_logger`, formatação de logs |

Não há pacote `application/services/llm` no motor ao vivo atual; decisão é exclusivamente Deep Learning quando `deep_learning.enabled` é verdadeiro.

## Dados e artefatos

| Caminho | Uso |
|---------|-----|
| `data/state.json` | Estado de contratos e banca (via `repo_path`) |
| `data/dl/{symbol}.pth` | Checkpoints PyTorch + calibrador + métricas (e.g., `R_50.pth`) |
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

Walk-forward DL:

```bash
cd app && conda run -n deriv-api python scripts/backtest/dl_walkforward.py --symbol R_50
```
