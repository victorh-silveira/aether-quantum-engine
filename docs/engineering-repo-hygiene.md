# Higienizacao do repositorio

Runbook para auditar e remover morto comprovado sem quebrar DEMO/SSOT.

Rule: `aether-repo-hygiene.mdc`. Skill: `aether-repo-hygiene`.
Deps pip: [`engineering-python-deps.md`](engineering-python-deps.md) + skill `aether-python-deps`.

## Objetivo

Eliminar codigo, scripts, containers, docs, skills/rules orfaos, JSON, git/Makefile e libs sem callers — com evidencia (grep/import/compose/matriz), nao por “pouco uso”.

## Ondas (ordem segura)

1. **Morto comprovado** — modulos/scripts sem import/Makefile/docs; lixo untracked (`.bak`); placeholders `.gitignore`; templates/workflows obsoletos; aliases Makefile.
2. **JSON / settings** — chaves top-level sem `resolve_*` / leitores em `app/src`; duplicatas flat vs nested SSOT; atualizar `test_doctrine_settings_ssot`.
3. **Alinhamentos** — scripts vivos com TF/seed legado; CHANGELOG `[Unreleased]` confuso; nomenclatura vs SSOT; hydrate/Docker coherente.
4. **Deps Python** — seguir skill `aether-python-deps` (nao inventar purge paralelo).
5. **QA** — `make app-pre-commit-run` (ou commit com hooks) apos cada onda; commits PT-BR separados (`repo` / `config` / `infra` / `deps`).

## Inventario obrigatorio antes de apagar

| Superficie | Como provar morto |
|------------|-------------------|
| Python `app/src` | Zero importers; Vulture; testes nao referenciam |
| Scripts | Sem Makefile/batch/`structure.md`/chamadores |
| Compose / containers | Ausente de `docker-compose` profiles `core,ml` e de `make docker-*` |
| Docs `.md` | Fora de `docs/README.md` e `docs/agent-coverage.md` |
| Skills / rules | Fora da matriz `agent-coverage` e da tabela `AGENTS.md` |
| JSON | Sem `resolve_*` / leitores; nao confundir `risk` top-level com `risk_management` nem `snapshot["risk"]` |
| Git | Placeholders gitignore; workflows que citam blocos inexistentes |
| Makefile | Targets sem `help`/docs e sem callers |

## Nunca remover (sem mandato explicito)

- Skills/rules/docs **indexados** na matriz `agent-coverage`
- Containers `redis` / `timescaledb` / `minio` / meta / loss
- Pacotes `__init__.py` vazios (marcadores de pacote DDD)
- Camadas hexagonais “por estética”
- Prefixo legado `m5`/`m15` em assinatura (intencional)
- MinIO / TorchScript path wired em `infra_factory`
- Caps de stake, settlement Redis, timeouts
- Historico versionado do CHANGELOG (so limpar `[Unreleased]` confuso)

## Cursor / VS Code

Revisar `.cursor/rules`, `.cursor/skills`, `.vscode` (se existir):

- Manter o que a matriz e `AGENTS.md` citam
- Apagar so orfaos nao indexados e sem referencia em docs
- Nao afrouxar `alwaysApply` de doutrina/engenharia

## Evidencia minima por delete

1. Grep de simbolos/path no repo
2. Confirmacao na matriz / Makefile / compose
3. Teste doctrine ou policy se mexer em settings/deps
4. Pre-commit verde

## Anti-padroes

Apagar camada DDD inteira; caçar “pouco usado” em DEMO; remover skill indexada; misturar onda JSON com refactor hexagonal; commit unico gigante sem QA por onda.
