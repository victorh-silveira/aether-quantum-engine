# Linters e qualidade

Configuração centralizada de hooks e release semântico (sem infra Kubernetes/Terraform).

| Arquivo | Uso |
|---------|-----|
| `git-hooks/` | Wrappers bash para WSL; `bash linters/git-hooks/install.sh` ou `make pre-commit` |
| `../.pre-commit-config.yaml` | Config única dos hooks (raiz do repo) |
| `commitlint.config.mjs` | Mensagens de commit (Conventional Commits) |
| `releaserc.json` | semantic-release no CI |

## Gates do pre-commit

| Hook | Ferramenta | Escopo |
|------|------------|--------|
| Qualidade | Ruff, Interrogate, Vulture, limite 300 linhas | `app/src`, `app/tests` |
| Testes | pytest + coverage | 100% em `app/src` (inclui Triton, mean-reversion, consensus penalty, session_target_bootstrap, stop_win_target) |
| Segurança | Bandit, pip-audit | dependências e código |
| Limpeza | caches e artefatos locais | workspace |

Documentação do projeto (`docs/`, `README.md`) não entra nos gates; alterações em `.md` não disparam pytest.

Os gates executam `app/scripts/operations/clean_workspace.py` com `cwd` em `app/`, usando o Conda `deriv-api` (`linters/git-hooks/bin/python` no WSL).

Comandos:

```bash
make lint          # estágio lint
make test          # pytest + cobertura
make pre-commit    # instala hooks
make pre-commit-run  # roda todos os hooks
```
