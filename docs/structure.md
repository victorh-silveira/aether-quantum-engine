# Estrutura do repositorio

Layout de software (sem Kubernetes/Terraform). Infra de nuvem fica fora deste escopo.

```
aether-quantum-engine/
├── app/                    # Codigo Python, testes e scripts operacionais
│   ├── src/                # Dominio, aplicacao, infraestrutura, apresentacao
│   ├── tests/              # Pytest (unit)
│   ├── scripts/            # backtest, monitor, operations, batch
│   ├── aether_paths.py     # Raiz app/ e repo/ (config, data, logs)
│   ├── run.py              # Entrada do motor ao vivo
│   ├── pyproject.toml
│   └── requirements*.txt
├── config/                 # settings.json (versionado)
├── docs/                   # Documentacao
├── linters/                # pre-commit, commitlint, semantic-release
├── .github/                # CI (lint, test, security, release)
├── run.py                  # Atalho para app/run.py
└── Makefile                # install, lint, test, run
```

## Camadas em `app/src`

| Pasta | Responsabilidade |
|-------|------------------|
| `application/services/llm` | Medallion, Gemini, macro, indicadores |
| `application/services/orchestrator` | Ciclo ao vivo, execucao, settlement |
| `domain` | Modelos e `risk_manager` |
| `infrastructure` | WebSocket Deriv, persistencia, handlers |
| `presentation` | Logger terminal |

## Dados e logs

Persistidos na raiz do repo (`data/`, `logs/`), referenciados via `aether_paths.repo_path()`.

## Comandos

```bash
make install
make test
make run
make backtest ARGS="--mode gemini --days 14"
```

Pre-commit (WSL): `make pre-commit` instala hooks bash em `.git/hooks`; depois `git commit` dispara lint/test/security.


