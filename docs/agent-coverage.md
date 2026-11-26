# Matriz de Cobertura de Agentes (100%)

Cada superficie tecnica do Aether Quantum Engine possui cobertura estrita formada por **documentacao tecnica + regra (.mdc) + habilidade (skill)**. O ponto de entrada soberano e o [`AGENTS.md`](../AGENTS.md).

O harness de agentes reside na pasta canonica [`.agents/`](../.agents/) e e versionado no Git de forma agnostica para qualquer IA/LLM.

A sincronizacao e testada automaticamente em `tests/unit/repo/test_agent_coverage_matrix.py`.

## Matriz

| Superficie | Doc | Rule (`.agents/rules/`) | Skill (`.agents/skills/`) |
|------------|-----|-------------------------|---------------------------|
| Arquitetura senior (host/DDD/asyncio/Laya/infra) | [engineering-architecture-senior.md](engineering-architecture-senior.md) + [arquitetura.md](arquitetura.md) | `aether-architecture-senior.mdc` | `aether-architecture-senior` |
| Motor continuo Laya System 1 (R_10 M5, zero SKIP) | [strategy-laya.md](strategy-laya.md) + [arquitetura.md](arquitetura.md) | `aether-laya-decision.mdc` | `aether-laya-engine` |
| Indicadores tecnicos causais unificados | [strategy-laya.md](strategy-laya.md) | `aether-laya-decision.mdc` + `aether-domain-pure.mdc` | `aether-laya-engine` |
| Runtime CPython 3.13 / GIL / GC / Tier2 | [engineering-python-313-runtime.md](engineering-python-313-runtime.md) | `aether-python-313-runtime.mdc` | `aether-python-313-runtime` |
| Asyncio TaskGroup / supervisor | [engineering-python-313-runtime.md](engineering-python-313-runtime.md) + [engineering-architecture-senior.md](engineering-architecture-senior.md) | `aether-python-313-runtime.mdc` | `aether-asyncio-supervisor` |
| Polars / Arrow zero-copy | [engineering-python-deps.md](engineering-python-deps.md) + [engineering-python-313-runtime.md](engineering-python-313-runtime.md) | `aether-python-deps.mdc` | `aether-polars-arrow` |
| Torch CUDA / Laya inferencia | [strategy-laya.md](strategy-laya.md) + [engineering-python-313-runtime.md](engineering-python-313-runtime.md) | `aether-laya-decision.mdc` | `aether-torch-cuda-infer` |
| asyncpg / Timescale | [infra-docker.md](infra-docker.md) + [engineering-devops-cloudops-senior.md](engineering-devops-cloudops-senior.md) | `aether-infra.mdc` | `aether-asyncpg-timescale` |
| Redis hiredis / settlement ZSET | [engineering-settlement.md](engineering-settlement.md) + [engineering-devops-cloudops-senior.md](engineering-devops-cloudops-senior.md) | `aether-settlement.mdc` | `aether-redis-hiredis` |
| Doutrina / sessao continua | [llm-trading-doctrine.md](llm-trading-doctrine.md) | `aether-llm-doctrine.mdc` | `aether-session-review` |
| Gestao de capital dinamica [1%, 5%] | [strategy-laya.md](strategy-laya.md) + [llm-trading-doctrine.md](llm-trading-doctrine.md) | `aether-risk-sizing.mdc` | `aether-session-review` |
| Orchestrator / ciclo M5 | [engineering-orchestrator.md](engineering-orchestrator.md) | `aether-orchestrator.mdc` | `aether-cycle-debug` |
| Settlement / Redis fila | [engineering-settlement.md](engineering-settlement.md) | `aether-settlement.mdc` | `aether-settlement-debug` |
| Infra Docker / state / storage / market | [infra-docker.md](infra-docker.md) | `aether-infra.mdc` | `aether-infra-stack` |
| DevOps / CloudOps senior (Compose, Redis, TS, MinIO) | [engineering-devops-cloudops-senior.md](engineering-devops-cloudops-senior.md) | `aether-infra.mdc` | `aether-devops-cloudops` |
| Deriv API / WS / PAT | [deriv-api-aether.md](deriv-api-aether.md) | `aether-deriv-api.mdc` | `aether-deriv-connect` |
| Settings / knobs SSOT | [engineering-settings-ssot.md](engineering-settings-ssot.md) | `aether-settings-ssot.mdc` | `aether-settings-change` |
| Engenharia / QA / testes 100% | [engineering-standards.md](engineering-standards.md) + [WORKFLOWS.md](../.github/workflows/WORKFLOWS.md) | `aether-engineering.mdc` + `aether-testing.mdc` | `aether-precommit` |
| Deps Python / requirements | [engineering-python-deps.md](engineering-python-deps.md) + [engineering-python-313-runtime.md](engineering-python-313-runtime.md) | `aether-python-deps.mdc` | `aether-python-deps` |
| Higienizacao do repositorio | [engineering-repo-hygiene.md](engineering-repo-hygiene.md) | `aether-repo-hygiene.mdc` | `aether-repo-hygiene` |
| Surface sync (fechamento) | [engineering-surface-sync.md](engineering-surface-sync.md) | `aether-surface-sync.mdc` | `aether-surface-sync` |
| Git commit padronizado / commitlint | [engineering-standards.md](engineering-standards.md) | `aether-git-commit.mdc` | `aether-git-commit` |
| Logging / telemetria (`INDICATORS` / `LAYA_DECISION`) | [engineering-observability.md](engineering-observability.md) + [engineering-logging-inventory.md](engineering-logging-inventory.md) | `aether-logging.mdc` | `aether-session-review` |
| Scripts / ops | [structure.md](structure.md) | `aether-scripts.mdc` | `aether-ops-runbook` |
| Domain models/indicators/risk puro | [structure.md](structure.md) | `aether-domain-pure.mdc` | — |

## Pastas DDD ↔ matriz

| Pasta | Linha da matriz |
|-------|-----------------|
| `src/` (camadas + ports) | Arquitetura senior (host/DDD/asyncio/Laya/infra) |
| `src/application/services/orchestrator/` | Orchestrator / ciclo M5 |
| `src/infrastructure/inference/` | Motor continuo Laya System 1 (R_10 M5, zero SKIP) |
| `src/domain/indicators/` | Indicadores tecnicos causais unificados |
| `src/domain/risk/` | Gestao de capital dinamica [1%, 5%] |
| `src/domain/models/`, `src/domain/math/`, `src/domain/symbols/` | Domain models/indicators/risk puro |
| `src/infrastructure/api/`, `src/infrastructure/websocket/` | Deriv API / WS / PAT |
| `src/infrastructure/state/`, `storage/`, `telemetry/` | Infra Docker / state / storage / market |
| `src/presentation/` | Logging / telemetria (`INDICATORS` / `LAYA_DECISION`) |
| `scripts/` | Scripts / ops |
| `tests/` | Engenharia / QA / testes 100% |
| `config/settings.json` | Settings / knobs SSOT |
| `infra/docker/` | Infra Docker / state / storage / market |

## Rules alwaysApply

Todas as rules em `.agents/rules/*.mdc` estao com `alwaysApply: true`.
