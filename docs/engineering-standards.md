# Engenharia e QA

Padroes obrigatorios para contribuicao e agentes. Entrada: [`AGENTS.md`](../AGENTS.md).

## Ambiente

- Python **3.13** (conda `deriv-api` / `.python-version`); motor **asyncio no host**; sidecars Docker `core,ml`
- Arquitetura DDD/hexagonal: [`engineering-architecture-senior.md`](engineering-architecture-senior.md)
- Comandos de terminal no **WSL**
- Codigo sem comentarios inline; docstrings OK
- Arquivos em `app/src/**/*.py` com no maximo **300 linhas**
- Event loop: nao bloquear com PyTorch/Polars/NumPy pesado no hot path (offload)

## Pre-commit

[`.pre-commit-config.yaml`](../.pre-commit-config.yaml) e as actions em [`.github/actions/ci/`](../.github/actions/ci/) chamam o mesmo motor `app/scripts/operations/clean_workspace.py --area --stage` (paridade bidirecional). Crash-first: lint → validate → security → test → build.

| Area | Stage | Pre-commit / CI |
|------|-------|-----------------|
| python | lint | Ruff, Interrogate, Vulture, 300 linhas |
| python | JSON / YAML | steps `Python \| JSON *` e `Python \| YAML *` (`--config-text json\|yaml`) |
| python | validate / build | compileall `app/src` |
| python | security | Bandit + pip-audit + Gitleaks (CI fail-closed) |
| python | test | pytest + cobertura **100%** em `app/src` |
| docker | lint…build | Hadolint, compose config, Trivy, smoke, `docker build` (build no CI) |
| shell | lint / validate | `bash -n` / shellcheck em `*.sh` |
| local | clean | `--stage clean --light-clean` |
| commit | commitlint | primeiro no pre-commit (`COMMIT_EDITMSG`) + hook `commit-msg` |

Rodar: `make app-pre-commit-run` ou `pre-commit run --all-files` (WSL, raiz do repo). Gitleaks precisa estar no PATH local. Detalhe: [`../linters/README.md`](../linters/README.md) e [`.github/README.md`](../.github/README.md).

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

Modulo `app/src/application/services/doctrine_invariants.py` tipa knobs da doutrina (edge floors, ACC, sample_size, caps, fusion/neg_edge). Testes:

- `tests/unit/application/test_doctrine_invariants.py`
- `tests/unit/application/test_doctrine_settings_ssot.py` (congela `settings.json` de producao)
- `tests/unit/repo/test_agent_coverage_matrix.py` (matriz ↔ `.cursor/rules|skills` + docs)
