---
name: aether-git-commit
description: Padrao obrigatorio de commits Git via Conventional Commits em PT-BR e hook commit-msg.
---

# Execução de Commits Padronizados Git

## 1. Contexto e Pré-requisitos
- Leia `AGENTS.md` e `.agents/rules/aether-git-commit.mdc`.
- Modificacoes devem estar devidamente testadas antes do commit.

## 2. Procedimento Operacional Passo a Passo
1. **Auditoria da Superfície:**
   - Verifique os arquivos modificados via `git status` e confira a integridade com `git diff`.
2. **Formatação da Mensagem:**
   - Estruture a mensagem no padrao Conventional Commits em Portugues:
     ```
     tipo(escopo): assunto em portugues no imperativo

     Corpo detalhado explicando o motivo da alteracao e o que foi realizado.
     ```
   - Utilize apenas escopos permitidos: `all`, `api`, `app`, `config`, `deps`, `domain`, `engine`, `infra`, `llm`, `orchestrator`, `pres`, `release`, `repo`, `risk`, `scripts`, `test`, `tools`, `ws`.
3. **Execução do Commit Oficial:**
   - Utilize o script oficial que garante pre-commit e commitlint:
     ```bash
     bash scripts/wsl/commit.sh "tipo(escopo): mensagem descritiva"
     # ou
     make app-commit MSG="tipo(escopo): mensagem descritiva"
     ```

## 3. Comandos de Verificação
```bash
git log -1 --stat
npx commitlint --from HEAD~1 --to HEAD --verbose
```

## 4. Critérios de Aceite
- Mensagem de commit 100% em PT-BR.
- Pre-commit e commit-msg executados com sucesso absoluto.
