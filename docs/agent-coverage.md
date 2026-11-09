# Matriz de Cobertura de Agentes (100%)

Cada superficie tecnica do Aether Quantum Engine possui cobertura estrita formada por **documentacao tecnica + regra (.mdc) + habilidade (skill)**. O ponto de entrada soberano e o [`AGENTS.md`](../AGENTS.md).

O harness de agentes reside na pasta canonica [`.agents/`](../.agents/) e e versionado no Git de forma agnostica para qualquer IA/LLM.

A sincronizacao e testada automaticamente em `tests/unit/repo/test_agent_coverage_matrix.py`.

## Matriz

| Superficie | Doc | Rule (`.agents/rules/`) | Skill (`.agents/skills/`) |
|------------|-----|-------------------------|---------------------------|
| Arquitetura senior (host/DDD/asyncio/ML/infra) | [engineering-architecture-senior.md](engineering-architecture-senior.md) + [arquitetura.md](arquitetura.md) | `aether-architecture-senior.mdc` | `aether-architecture-senior` |
| Runtime CPython 3.13 / GIL / GC / Tier2 | [engineering-python-313-runtime.md](engineering-python-313-runtime.md) | `aether-python-313-runtime.mdc` | `aether-python-313-runtime` |
| Asyncio TaskGroup / supervisor | [engineering-python-313-runtime.md](engineering-python-313-runtime.md) + [engineering-architecture-senior.md](engineering-architecture-senior.md) | `aether-python-313-runtime.mdc` | `aether-asyncio-supervisor` |
| Polars / Arrow zero-copy | [engineering-python-deps.md](engineering-python-deps.md) + [engineering-python-313-runtime.md](engineering-python-313-runtime.md) | `aether-python-deps.mdc` | `aether-polars-arrow` |
| Torch CUDA inferencia | [engineering-deep-learning.md](engineering-deep-learning.md) + [engineering-python-313-runtime.md](engineering-python-313-runtime.md) | `aether-deep-learning.mdc` | `aether-torch-cuda-infer` |
| asyncpg / Timescale | [infra-docker.md](infra-docker.md) + [engineering-devops-cloudops-senior.md](engineering-devops-cloudops-senior.md) | `aether-infra.mdc` | `aether-asyncpg-timescale` |
| Redis hiredis / settlement ZSET | [engineering-settlement.md](engineering-settlement.md) + [engineering-devops-cloudops-senior.md](engineering-devops-cloudops-senior.md) | `aether-settlement.mdc` | `aether-redis-hiredis` |
| Doutrina / sessao | [llm-trading-doctrine.md](llm-trading-doctrine.md) | `aether-llm-doctrine.mdc` | `aether-session-review` |
| Playbook senior binario | [binary-senior-playbook.md](binary-senior-playbook.md) | `aether-execution-gates.mdc` | `aether-binary-senior` |
| Risco / Kelly | [medallion.md](medallion.md) + [llm-trading-doctrine.md](llm-trading-doctrine.md) | `aether-risk-sizing.mdc` | `aether-session-review` |
| Execution gates (inversao condicionada a edge e proposta cotada) | [binary-senior-playbook.md](binary-senior-playbook.md) + [engineering-settings-ssot.md](engineering-settings-ssot.md) + [engineering-indicator-gates.md](engineering-indicator-gates.md) | `aether-execution-gates.mdc` | `aether-session-review` + `aether-binary-senior` |
| Gates por indicadores (catalogo / backlog) | [engineering-indicator-gates.md](engineering-indicator-gates.md) | `aether-execution-gates.mdc` | `aether-binary-senior` + `aether-session-review` |
| Compra por setup M15 do indicador | [arquitetura.md](arquitetura.md) + [engineering-settings-ssot.md](engineering-settings-ssot.md) | `aether-indicator-consensus.mdc` | `aether-indicator-consensus` + `aether-settings-change` |
| Setup M15 operacional e pesquisa H1/D1 | [strategy-pullback.md](strategy-pullback.md) | `aether-indicator-consensus.mdc` | `aether-indicator-consensus` |
| Fusao EV multi-escala | [binary-senior-playbook.md](binary-senior-playbook.md) + [engineering-orchestrator.md](engineering-orchestrator.md) | `aether-execution-gates.mdc` | `aether-binary-senior` + `aether-cycle-debug` |
| Loss-classifier (pesquisa legada) | [infra-docker.md](infra-docker.md) | `aether-execution-gates.mdc` + `aether-infra.mdc` | `aether-infra-stack` |
| Sample size / SIDE_EQ | [sample-size-lln.md](sample-size-lln.md) | `aether-sample-size.mdc` | `aether-session-review` |
| Orchestrator / ciclo | [engineering-orchestrator.md](engineering-orchestrator.md) | `aether-orchestrator.mdc` | `aether-cycle-debug` |
| Scale vision MACRO/MICRO/MINI/MILI | [engineering-orchestrator.md](engineering-orchestrator.md) + [binary-senior-playbook.md](binary-senior-playbook.md) | `aether-execution-gates.mdc` | `aether-cycle-debug` + `aether-binary-senior` |
| DL / labels / calib / vies de classe / alinhamento M5 fechado na inferencia | [engineering-deep-learning.md](engineering-deep-learning.md) | `aether-deep-learning.mdc` | `aether-dl-train` |
| Settlement / Redis fila | [engineering-settlement.md](engineering-settlement.md) | `aether-settlement.mdc` | `aether-settlement-debug` |
| Infra Docker / state / storage / market / inference | [infra-docker.md](infra-docker.md) | `aether-infra.mdc` | `aether-infra-stack` |
| DevOps / CloudOps sênior (Compose, Redis, TS, MinIO) | [engineering-devops-cloudops-senior.md](engineering-devops-cloudops-senior.md) | `aether-infra.mdc` | `aether-devops-cloudops` |
| Deriv API / WS / PAT | [deriv-api-aether.md](deriv-api-aether.md) | `aether-deriv-api.mdc` | `aether-deriv-connect` |
| Settings / knobs SSOT | [engineering-settings-ssot.md](engineering-settings-ssot.md) | `aether-settings-ssot.mdc` | `aether-settings-change` |
| Engenharia / QA / testes | [engineering-standards.md](engineering-standards.md) + [WORKFLOWS.md](../.github/workflows/WORKFLOWS.md) | `aether-engineering.mdc` + `aether-testing.mdc` | `aether-precommit` |
| Deps Python / requirements | [engineering-python-deps.md](engineering-python-deps.md) + [engineering-python-313-runtime.md](engineering-python-313-runtime.md) | `aether-python-deps.mdc` | `aether-python-deps` |
| Higienizacao do repositorio | [engineering-repo-hygiene.md](engineering-repo-hygiene.md) | `aether-repo-hygiene.mdc` | `aether-repo-hygiene` |
| Surface sync (fechamento) | [engineering-surface-sync.md](engineering-surface-sync.md) | `aether-surface-sync.mdc` | `aether-surface-sync` |
| Git commit padronizado / commitlint | [engineering-standards.md](engineering-standards.md) | `aether-git-commit.mdc` | `aether-git-commit` |
| Contrato prompt-modelo | [prompt-model.md](../prompt-model.md) | `aether-engineering.mdc` | `aether-surface-sync` |
| Logging / presentation (telemetria de contrato e Brier auditado) | [engineering-observability.md](engineering-observability.md) + [engineering-logging-inventory.md](engineering-logging-inventory.md) | `aether-logging.mdc` | `aether-session-review` |
| Scripts / ops | [structure.md](structure.md) | `aether-scripts.mdc` | `aether-ops-runbook` |
| Domain models/math/symbols | [structure.md](structure.md) + [deriv-indices-algorithm.md](deriv-indices-algorithm.md) | `aether-domain-pure.mdc` | — |
| Volatility 75 (1s) Index | [deriv-indices-algorithm.md](deriv-indices-algorithm.md) | `aether-v75-market.mdc` | `aether-v75-market-analyst` |
| Sizing Single-Strike 4.31% / Payout 0.85 | [medallion.md](medallion.md) + [llm-trading-doctrine.md](llm-trading-doctrine.md) | `aether-risk-sizing.mdc` | `aether-session-review` |
| Indicador publico de sinais sinteticos | [strategy-pullback.md](strategy-pullback.md) + [arquitetura.md](arquitetura.md) | `aether-indicator.mdc` | `aether-indicator` |
| Verificador de Sinais M15 | [binary-senior-playbook.md](binary-senior-playbook.md) | `aether-execution-gates.mdc` | `aether-binary-senior` |

## Pastas DDD ↔ matriz

| Pasta | Linha da matriz |
|-------|-----------------|
| `src/` (camadas + ports) | Arquitetura senior (host/DDD/asyncio/ML/infra) |
| `src/application/services/orchestrator/` | Orchestrator / ciclo |
| `src/application/services/execution_scale_*.py` | Scale vision telemetria / regime (sem alterar lado) |
| `src/application/services/deep_learning/` | DL / labels / calib / vies de classe |
| `src/application/services/execution_*.py` | Execution: tecnico + confluencia institucional |
| `src/domain/risk/` | Risco / Kelly |
| `src/domain/analytics/` | Sample size / SIDE_EQ |
| `src/domain/models/`, `domain/math/`, `domain/symbols/` | Domain models/math/symbols |
| `src/infrastructure/api/`, `infrastructure/websocket/` | Deriv API |
| `src/infrastructure/state/`, `storage/`, `telemetry/` | Infra |
| `src/presentation/` | Logging |
| `scripts/` | Scripts / ops |
| `app/tests/` | Engenharia / QA / testes |
| `config/settings.json` | Settings SSOT |
| `infra/docker/` | Infra |

## Rules alwaysApply

Todas as rules em `.agents/rules/*.mdc` estao com `alwaysApply: true` (doutrina, engenharia, dominio, ops, deps, higiene, surface sync).
