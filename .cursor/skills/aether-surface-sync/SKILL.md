---
name: aether-surface-sync
description: >-
  Fecha mudancas Aether sincronizando docs/rules/skills/AGENTS/matriz, rodando
  pre-commit no WSL e removendo codigo morto ou sujeira local. Use when finishing
  a feature, settings change, infra/docs update, or when the user mentions sync
  de superficie, fechar PR, atualizar agents, ou checklist pos-mudanca.
---

# Surface sync (fechamento de mudanca)

## Quando aplicar

Apos qualquer mudanca material no motor, settings, Docker, doutrina, gates, deps ou superficie Cursor — **antes** do commit/push final.

## Checklist obrigatorio

1. **Escopo** — listar arquivos tocados e superficies da matriz `docs/agent-coverage.md` impactadas
2. **Docs** — atualizar `.md` de engenharia/ops/doutrina cujo significado operacional mudou (nao so o codigo)
3. **Rules** — se knob/gate/contrato/SSOT mudou, alinhar `.cursor/rules/*.mdc` (`alwaysApply: true`)
4. **Skills** — se o procedimento do agente mudou, alinhar `.cursor/skills/*/SKILL.md`
5. **Indices** — `AGENTS.md` (tabela Leitura por tarefa) + linha em `docs/agent-coverage.md`; skill/rule nova = entrada na matriz
6. **Contrato** — se mudar padrao DDD/QA/DX cross-repo, atualizar `prompt-model.md`
7. **Anti-sujeira** — apagar `_tmp*`, `COMMIT_MSG.txt`, probes locais; grep por refs stale ao path/simbolo removido; sem imports mortos
8. **Higiene pontual** — se houver morto comprovado no diff, seguir `aether-repo-hygiene` (evidencia antes de delete)
9. **Pre-commit** — WSL + Conda `deriv-api`: `make app-pre-commit-run` (ou commit com hooks). Falha → skill `aether-precommit`
10. **Entrega** — commits PT-BR; nao deixar superficie agentes desatualizada “para depois”

## Nunca

- Commitar so codigo e deixar doutrina/rules/skills mentindo o SSOT
- Pular pre-commit ou baixar `cov-fail-under` / limite 300 linhas
- Apagar skill/doc indexado na matriz sem atualizar `agent-coverage` + `AGENTS.md`
- Deixar artefatos `_tmp*` / backups no working tree

## Docs

`docs/engineering-surface-sync.md`, `docs/agent-coverage.md`, `AGENTS.md`, `prompt-model.md`, skills `aether-precommit` + `aether-repo-hygiene`
