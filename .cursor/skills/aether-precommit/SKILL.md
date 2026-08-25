---
name: aether-precommit
description: >-
  Diagnostica e corrige falhas de pre-commit do Aether (Ruff, Interrogate,
  Vulture, limite 300 linhas, pytest cobertura 100%, Bandit, Gitleaks, commitlint PT-BR).
  Use when pre-commit fails, coverage is below 100%, a file exceeds 300 lines,
  or the user mentions clean_workspace, cov-fail-under, or commitlint.
---

# Pre-commit Aether

## Passos

1. Ler o trecho FAIL do hook (lint / test / security / commitlint)
2. Lint: Ruff format/fix; Interrogate docstrings publicas; Vulture dead code; split se >300 linhas
3. Test: rodar o teste falho; adicionar cobertura dos misses reportados
4. Security: corrigir Bandit real; Gitleaks no stage `security` via `clean_workspace` (mesmo gate do CI; binario no PATH); nao `# nosec` sem justificativa forte
5. Commitlint: tipo+escopo validos; assunto PT-BR; corpo nao vazio
6. Reexecutar `pre-commit run --all-files` no **WSL**

## Docs

`docs/engineering-standards.md`, `AGENTS.md`
