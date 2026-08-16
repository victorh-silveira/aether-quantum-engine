# Linters e qualidade

Configuracao centralizada de hooks e release semantico.

| Arquivo | Uso |
|---------|-----|
| `git-hooks/` | Wrappers bash para WSL; `bash linters/git-hooks/install.sh` ou `make app-pre-commit` |
| `../.pre-commit-config.yaml` | Config unica dos hooks (raiz do repo) |
| `commitlint.config.mjs` | Mensagens de commit (Conventional Commits) |
| `releaserc.json` | semantic-release no CI |

## Gates do pre-commit

| Hook | Ferramenta | Escopo |
|------|------------|--------|
| Qualidade | Ruff, Interrogate, Vulture, limite 300 linhas | `app/src`, `app/tests` |
| Testes | pytest-xdist + coverage | 100% em `app/src` (**305** arquivos de teste) |
| Seguranca | Bandit, pip-audit | dependencias e codigo |
| Limpeza | caches e artefatos locais | workspace |

Documentacao (`docs/`, `README.md`) nao entra nos gates; alteracoes em `.md` nao disparam pytest.

Os gates executam `app/scripts/operations/clean_workspace.py` com `cwd` em `app/`, usando o Conda `deriv-api` (`linters/git-hooks/bin/python` no WSL).

Comandos:

```bash
make app-lint
make app-test
make app-pre-commit
make app-pre-commit-run
```

Verificacao estrutural: maximo **300 linhas** por arquivo em `app/src/` (estagio lint).
