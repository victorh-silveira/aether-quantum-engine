---
name: aether-precommit
description: Execucao e correcao de falhas de pre-commit, Ruff, Interrogate, Vulture e cobertura 100%.
---

# Execução e Validação de Pre-commit

## 1. Contexto e Pré-requisitos
- Leia `docs/engineering-standards.md` e `.agents/rules/aether-testing.mdc`.

## 2. Procedimento Operacional Passo a Passo
1. **Execução do Pipeline Completo:**
   - Execute o pipeline unificado de limpeza e verificacao:
     ```bash
     make app-lint
     make app-test
     ```
2. **Correção de Falhas:**
   - **Ruff:** execute `/mnt/c/Users/victor-silveira/anaconda3/envs/deriv-api/python.exe -m ruff check --fix app/src`.
   - **Docstrings:** adicione docstrings descritivas em modulos, classes e metodos apontados pelo Interrogate.
   - **Vulture:** remova codigo morto nao referenciado.
   - **Limite 300 linhas:** decomponha arquivos extensos em submódulos coesos.

## 3. Comandos de Verificação
```bash
make app-lint
make app-test
```

## 4. Critérios de Aceite
- Zero erros de lint, 100% de docstrings, zero codigo morto e 100% de cobertura de testes.
