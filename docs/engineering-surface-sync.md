# Surface sync (fechamento de mudanca)

Toda atualizacao material do Aether deve fechar com superficie de agentes coerente, pre-commit verde e working tree limpo.

Rule: `aether-surface-sync.mdc`. Skill: `aether-surface-sync`.

Complementa:

- [`engineering-repo-hygiene.md`](engineering-repo-hygiene.md) — purge de morto comprovado
- [`engineering-standards.md`](engineering-standards.md) — QA / cobertura 100%
- Skill `aether-precommit` — diagnostico de hook vermelho

## O que sincronizar

| Mudanca | Superficie tipica |
|---------|-------------------|
| Arquitetura / camadas / event loop / host | `engineering-architecture-senior` + rule/skill `aether-architecture-senior` + `arquitetura.md` |
| Knob / settings | doc settings SSOT + rule `aether-settings-ssot` + skill `aether-settings-change` |
| Gate / risco / fusao | playbook + doutrina + rules de execucao/risco |
| DL / horizon / label | `engineering-deep-learning` + rule/skill DL |
| Docker / meta / loss | `infra-docker` + rule/skill infra |
| Nova skill ou rule | `docs/agent-coverage.md` + tabela em `AGENTS.md` |
| Deps pip | `engineering-python-deps` + rule/skill deps |
| Contrato de engenharia / scaffold | `prompt-model.md` (raiz) + rule `aether-engineering` |

## Ordem de fechamento

1. Codigo + testes da mudanca
2. Docs / rules / skills / `AGENTS.md` / `agent-coverage.md`
3. Remover sujeira local (`_tmp*`, msgs de commit soltas, probes)
4. `make app-pre-commit-run` no WSL (Conda `deriv-api`)
5. Commit PT-BR (hooks ativos)

## Criterio de pronto

- Matriz `agent-coverage` aponta para arquivos existentes
- Nenhuma rule com `alwaysApply: false`
- Pre-commit (lint + test cov 100% + security + cleanup) verde
- Sem artefatos temporarios no diff a publicar
