# Engenharia e QA

Padroes obrigatorios para contribuicao e agentes. Entrada: [`AGENTS.md`](../AGENTS.md).

## Ambiente

- Python **3.13** (conda `deriv-api` / `.python-version`)
- Comandos de terminal no **WSL**
- Codigo sem comentarios inline; docstrings OK
- Arquivos em `app/src/**/*.py` com no maximo **300 linhas**

## Pre-commit

[`.pre-commit-config.yaml`](../.pre-commit-config.yaml) chama `app/scripts/operations/clean_workspace.py`:

| Stage | O que faz |
|-------|-----------|
| lint | Ruff, Interrogate, Vulture, limite de linhas |
| test | pytest + cobertura **100%** em `app/src` |
| security | Bandit + pip-audit |
| cleanup | caches/artefatos locais |
| commit-msg | commitlint (`linters/commitlint.config.mjs`) |

Rodar: `pre-commit run --all-files` (no WSL, a partir da raiz do repo).

## Commitlint

- Assunto em **PT-BR** (plugin bloqueia verbos ingles comuns)
- Escopo obrigatorio: `all`, `api`, `app`, `config`, `deps`, `domain`, `engine`, `infra`, `llm`, `orchestrator`, `pres`, `release`, `repo`, `risk`, `scripts`, `test`, `tools`, `ws`
- Corpo do commit obrigatorio

Exemplo:

```
fix(engine): endurece piso de margem calibrada

Bloqueia explore com Cal fraco e edge negativo.
```

## Testes

- Layout: `app/tests/unit/{application,domain,infrastructure,presentation,scripts}/`
- Espelhar a camada do codigo sob teste
- Cobertura fail-under **100%** em `src`
- Novos ramos: teste unitario no mesmo PR

## Fluxo de contribuicao

1. Ler [`AGENTS.md`](../AGENTS.md) e a linha correspondente em [`agent-coverage.md`](agent-coverage.md)
2. Mudanca minima; hipotese falsificavel se for knob
3. Pre-commit verde
4. Commit PT-BR + push (so se pedido)

Sem `CONTRIBUTING.md` separado: este doc + AGENTS sao o SSOT de contribuicao.

## Enforcement da doutrina no core

Modulo `app/src/application/services/doctrine_invariants.py` tipa knobs da doutrina (`hard_cal_margin_floor`, edge floors, ACC, sample_size, caps). Testes:

- `tests/unit/application/test_doctrine_invariants.py`
- `tests/unit/application/test_doctrine_settings_ssot.py` (congela `settings.json` de producao)
- `tests/unit/repo/test_agent_coverage_matrix.py` (matriz ↔ `.cursor/rules|skills` + docs)
