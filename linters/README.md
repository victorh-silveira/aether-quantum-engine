# Linters e qualidade

Configuracao centralizada de hooks e release semantico.

| Arquivo | Uso |
|---------|-----|
| `git-hooks/` | Wrappers bash para WSL; `bash linters/git-hooks/install.sh` ou `make app-pre-commit` |
| `../.pre-commit-config.yaml` | Config unica dos hooks (raiz do repo) |
| `commitlint.config.mjs` | Mensagens de commit (Conventional Commits) |
| `releaserc.json` | semantic-release no CI |

## Gates do pre-commit

Crash-first: **commitlint** → python (codigo + JSON + YAML) → docker → shell (Lint, Validate, Seguranca, Testes, Build) → limpeza.

| Hook | Ferramenta | Escopo |
|------|------------|--------|
| Commitlint | Conventional Commits PT-BR | primeiro (COMMIT_EDITMSG + commit-msg) |
| Python | Ruff, Interrogate, Vulture, pytest 100%, Bandit | `app/` |
| Python \| JSON | json.loads, SSOT, `python.json` | `config/` |
| Python \| YAML | estrutura + actionlint | compose, pre-commit, `.github` |
| Docker | Hadolint, compose, Trivy | `infra/docker` (build so no CI) |
| Shell | shellcheck / `bash -n` | `*.sh` |
| Limpeza | caches | workspace |

Documentacao (`docs/`, `README.md`) nao entra nos gates Python de codigo; JSON/YAML do job Python cobrem `config/` e workflows.

Os gates executam `app/scripts/operations/clean_workspace.py --area --stage` com `cwd` em `app/` no Python, usando o Conda `deriv-api` (`linters/git-hooks/bin/python` no WSL).

Comandos:

```bash
make app-lint
make app-test
make app-pre-commit
make app-pre-commit-run
```

Verificacao estrutural: maximo **300 linhas** por arquivo em `app/src/` (estagio lint).
