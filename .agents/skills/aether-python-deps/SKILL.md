---
name: aether-python-deps
description: Gestao e auditoria de dependencias Python em requirements.txt e requirements-dev.txt.
---

# Auditoria e Gestão de Dependências Python

## 1. Contexto e Pré-requisitos
- Leia `docs/engineering-python-deps.md` e `.agents/rules/aether-python-deps.mdc`.

## 2. Procedimento Operacional Passo a Passo
1. **Inspeção de Locks:**
   - Inspecione `app/requirements.txt` e `app/requirements-dev.txt` para assegurar que cada pacote possua versao fixada (`==`).
2. **Verificação de Proibições:**
   - Rode o teste de politica de dependencias para validar a ausencia de bibliotecas desautorizadas (como pandas).
3. **Instalação Limpa:**
   - Execute a atualizacao do ambiente de forma controlada:
     ```bash
     make app-install
     ```

## 3. Comandos de Verificação
```bash
/mnt/c/Users/victor-silveira/anaconda3/envs/deriv-api/python.exe -m pytest app/tests/unit/repo/test_python_deps_policy.py -v
```

## 4. Critérios de Aceite
- Teste de politicas de dependencias passando sem advertencias.
