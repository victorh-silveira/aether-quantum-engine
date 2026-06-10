# Estrutura do repositório

Layout de software (sem infraestrutura de nuvem neste repo).

```
aether-quantum-engine/
├── app/
│   ├── src/
│   │   ├── application/services/
│   │   │   ├── deep_learning/     # TCN, treino deferido, gating, deploy, decision_bridge
│   │   │   ├── orchestrator/      # Ciclo, fases treino/operação, execução, settlement
│   │   │   ├── execution_*.py     # Direção, ranking de mercado, seleção obrigatória e recovery
│   │   │   ├── log_dedupe.py      # Deduplicação de logs repetidos
│   │   │   └── auth_manager.py
│   │   ├── domain/                # Modelos, risk_manager, martingale, stake
│   │   ├── infrastructure/        # WebSocket, stream, trade, persistência
│   │   └── presentation/          # Logger terminal
│   ├── tests/unit/                # Pytest (cobertura 100% em src)
│   ├── scripts/
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
| `application/services/deep_learning` | Features (matriz pré-computada), TCN, treino walk-forward deferido (prioridade bootstrap por sessão), calibração, predição, deploy gate, gate de treinamento, progresso por época |
| `application/services/orchestrator` | `Orchestrator`, `ExecutionManager` (fases treino/operação), `execution_collect`, `execution_blockers`, settlement, `post_settlement_cycle` |
| `application/services` | `execution_direction`, `execution_direction_fallback`, `execution_market_rank`, `execution_mandatory_pick`, `execution_symbols`, `execution_symbols_recovery`, `log_dedupe`, `auth_manager` |
| `domain` | `Candle`, `Trade`, `RiskManager`, Kelly, martingale, cooldowns, `stake_sizing` |
| `infrastructure` | `WebSocketManager`, `StreamHandler`, `TradeHandler`, `PersistenceManager` |
| `presentation/terminal` | `setup_logger`, `BlankLineSquasher`, formatação de logs |

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
