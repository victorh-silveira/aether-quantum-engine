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

1. Ler o trecho FAIL do hook (commitlint primeiro; depois `Python | …`, `Docker | …`, `Shell | …`)
2. Lint: Ruff format/fix; Interrogate docstrings publicas; Vulture dead code; split se >300 linhas; bloquear artefatos grandes (`.pt`/`.parquet`)
3. Validate: compileall (python); JSON/YAML em steps do job Python (`--config-text json|yaml`); compose/Hadolint na stack docker; shellcheck/`bash -n` na stack shell
4. Test: rodar o teste falho; adicionar cobertura dos misses reportados (linhas e branches; domain sem I/O)
5. Security: corrigir Bandit real; Gitleaks no stage `security` python (CI fail-closed; local aviso se binario ausente); nao `# nosec` sem justificativa forte; pip-audit CVE HIGH/CRITICAL
6. Commitlint: tipo+escopo validos; assunto PT-BR; corpo nao vazio — falha antes do pytest quando ha `COMMIT_EDITMSG`
7. Reexecutar `pre-commit run --all-files` no **WSL**

## Docs

`docs/engineering-standards.md`, `docs/engineering-architecture-senior.md`, `.github/README.md`, `AGENTS.md`
