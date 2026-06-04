# Linters e qualidade

Configuracao centralizada de hooks e release semantico (sem infra Kubernetes/Terraform).

| Arquivo | Uso |
|---------|-----|
| `git-hooks/` | Wrappers bash para WSL; `bash linters/git-hooks/install.sh` ou `make pre-commit` |
| `pre-commit-config.yaml` | Config dos hooks; apontada pelos wrappers em `git-hooks/` |
| `commitlint.config.mjs` | Mensagens de commit (Conventional Commits) |
| `releaserc.json` | semantic-release no CI |

Os gates executam `app/scripts/operations/clean_workspace.py` com `cwd` implicito em `app/` (Ruff, pytest, bandit, interrogate), usando o Conda `deriv-api` (`linters/git-hooks/bin/python` no WSL).
