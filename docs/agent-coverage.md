# Matriz de agentes

| Superfície | Código | Documento | Regra / skill |
|---|---|---|---|
| Catálogo e candles públicos | `app/src/indicator/deriv.py`, `domain.py` | [Arquitetura](arquitetura.md) | `aether-indicator` |
| Treino e modelos | `app/src/indicator/model.py`, `training.py` | [Modelos](modelos.md) | `aether-indicator` |
| Sinal e persistência | `app/src/indicator/service.py`, `storage.py` | [Arquitetura](arquitetura.md) | `aether-indicator` |
| Mercado em andamento | `app/src/indicator/domain.py`, `service.py`, `metrics.py`, `storage.py` | [Operação](operacao.md) | `aether-indicator` |
| Grafana e Prometheus | `infra/docker/` | [Operação](operacao.md) | `aether-indicator` |
| CI/CD e testes | `.github/workflows/ci.yml`, `app/tests/` | [Operação](operacao.md) | `aether-indicator` |

A regra versionada está em `.cursor/rules/aether-indicator.mdc` e a skill em `.agents/skills/aether-indicator/SKILL.md`. `AGENTS.md` é o índice do produto.
