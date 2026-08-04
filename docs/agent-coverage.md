# Matriz de cobertura do agente (100%)

Cada superficie do bot tem **doc + rule + skill** (ou `—` justificado). Entrada: [`AGENTS.md`](../AGENTS.md).

Rules/skills vivem em [`.cursor/`](../.cursor/) e sao **versionadas** no git.

Enforcement no core: `doctrine_invariants.py` + testes `test_doctrine_*` / `test_agent_coverage_matrix` (ver [engineering-standards.md](engineering-standards.md)).

## Matriz

| Superficie | Doc | Rule (`.cursor/rules/`) | Skill (`.cursor/skills/`) |
|------------|-----|-------------------------|---------------------------|
| Doutrina / sessao | [llm-trading-doctrine.md](llm-trading-doctrine.md) | `aether-llm-doctrine.mdc` | `aether-session-review` |
| Playbook senior binario | [binary-senior-playbook.md](binary-senior-playbook.md) | `aether-execution-gates.mdc` | `aether-binary-senior` |
| Risco / Kelly | [medallion.md](medallion.md) + doutrina | `aether-risk-sizing.mdc` | `aether-session-review` |
| Execution gates | [arquitetura.md](arquitetura.md) §6 | `aether-execution-gates.mdc` | `aether-session-review` |
| Sample size / SIDE_EQ | [sample-size-lln.md](sample-size-lln.md) | `aether-sample-size.mdc` | `aether-session-review` |
| Orchestrator / ciclo | [engineering-orchestrator.md](engineering-orchestrator.md) | `aether-orchestrator.mdc` | `aether-cycle-debug` |
| Scale vision MACRO/MICRO/MINI/MILI | [engineering-orchestrator.md](engineering-orchestrator.md) + [binary-senior-playbook.md](binary-senior-playbook.md) | `aether-execution-gates.mdc` | `aether-cycle-debug` + `aether-binary-senior` |
| DL / labels / calib / vies de classe | [engineering-deep-learning.md](engineering-deep-learning.md) | `aether-deep-learning.mdc` | `aether-dl-train` |
| Settlement / Redis fila | [engineering-settlement.md](engineering-settlement.md) | `aether-settlement.mdc` | `aether-settlement-debug` |
| Infra Docker / state / storage / market / inference | [infra-docker.md](infra-docker.md) | `aether-infra.mdc` | `aether-infra-stack` |
| Deriv API / WS / PAT | [deriv-api-aether.md](deriv-api-aether.md) | `aether-deriv-api.mdc` | `aether-deriv-connect` |
| Settings / knobs SSOT | [engineering-settings-ssot.md](engineering-settings-ssot.md) | `aether-settings-ssot.mdc` | `aether-settings-change` |
| Engenharia / QA / testes | [engineering-standards.md](engineering-standards.md) | `aether-engineering.mdc` + `aether-testing.mdc` | `aether-precommit` |
| Logging / presentation | [engineering-observability.md](engineering-observability.md) + [engineering-logging-inventory.md](engineering-logging-inventory.md) | `aether-logging.mdc` | `aether-session-review` |
| Scripts / ops | [structure.md](structure.md) §Scripts | `aether-scripts.mdc` | `aether-ops-runbook` |
| Domain models/math/symbols | [structure.md](structure.md) §Domain | `aether-domain-pure.mdc` | — |

## Pastas DDD ↔ matriz

| Pasta | Linha da matriz |
|-------|-----------------|
| `app/src/application/services/orchestrator/` | Orchestrator / ciclo |
| `app/src/application/services/execution_scale_*.py` | Scale vision + adaptacao de fita (lado/sizing) |
| `app/src/application/services/deep_learning/` | DL / labels / calib / vies de classe (sample_weighting, majority-collapse, regime via recency; `raw_extreme`) |
| `app/src/application/services/execution_*.py` | Execution gates |
| `app/src/domain/risk/` | Risco / Kelly |
| `app/src/domain/analytics/` | Sample size / SIDE_EQ |
| `app/src/domain/models|math|symbols/` | Domain models/math/symbols |
| `app/src/infrastructure/api|handlers stream|ws` | Deriv API |
| `app/src/infrastructure/state|storage|market|inference|factories` | Infra |
| `app/src/presentation/` | Logging |
| `app/scripts/` | Scripts / ops |
| `app/tests/` | Engenharia / QA / testes |
| `config/settings.json` | Settings SSOT |
| `infra/docker/` | Infra |

## Rules alwaysApply

- `aether-llm-doctrine.mdc` — processo > P&L
- `aether-engineering.mdc` — QA/WSL/300 linhas/cobertura/commitlint
