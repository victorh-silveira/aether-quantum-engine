---
name: aether-repo-hygiene
description: Higienizacao de codigo morto, branch cleanup, remocao de arquivos orfaos e auditoria Vulture.
---

# Higienização e Manutenção do Repositório

## 1. Contexto e Pré-requisitos
- Leia `docs/engineering-repo-hygiene.md` e `.agents/rules/aether-repo-hygiene.mdc`.

## 2. Procedimento Operacional Passo a Passo
1. **Detecção de Código Morto:**
   - Execute o Vulture para localizar variaveis, metodos ou classes sem utilizacao:
     ```bash
     python app/scripts/operations/clean_workspace.py --area python --stage lint
     ```
2. **Sanitização de Cache e Temporários:**
   - Limpe caches e arquivos de execucao local:
     ```bash
     make app-clean
     ```
3. **Auditoria da Árvore Git:**
   - Inspecione arquivos nao rastreados ou pastas vazias com `git status`.

## 3. Comandos de Verificação
```bash
git status --porcelain
make app-lint
```

## 4. Critérios de Aceite
- Zero codigo morto em `app/src/`.
- Working tree do Git estritamente limpa.
