---
name: aether-surface-sync
description: Protocolo obrigatorio de sincronizacao de superficie (docs, rules, skills, AGENTS.md).
---

# Sincronização de Superfície de IA e Engenharia

## 1. Contexto e Pré-requisitos
- Leia `docs/engineering-surface-sync.md`, `AGENTS.md` e `.agents/rules/aether-surface-sync.mdc`.

## 2. Procedimento Operacional Passo a Passo
1. **Mapeamento do Impacto:**
   - Identifique todos os documentos em `docs/`, regras em `.agents/rules/` e skills em `.agents/skills/` que foram afetados pela alteracao de codigo.
2. **Atualização Coordenada:**
   - Atualize os textos tecnicos eliminando divergencias conceituais.
   - Sincronize a matriz `docs/agent-coverage.md`.
3. **Validação Automatizada:**
   - Execute o teste de rastreabilidade de superficie:
     ```bash
     /mnt/c/Users/victor-silveira/anaconda3/envs/deriv-api/python.exe -m pytest tests/unit/repo/test_agent_coverage_matrix.py -v
     ```
4. **Fechamento via Commit Padronizado:**
   - Execute o commit oficial em PT-BR:
     ```bash
     make app-commit ARGS='docs repo "sincroniza superficie de agentes" "Alinha documentacao, regras e habilidades ao contrato vigente."'
     ```

## 3. Comandos de Verificação
```bash
/mnt/c/Users/victor-silveira/anaconda3/envs/deriv-api/python.exe -m pytest tests/unit/repo/test_agent_coverage_matrix.py -v
make app-lint
```

## 4. Critérios de Aceite
- 100% das regras e skills sincronizadas com a documentacao e validadas pelo teste automatizado.
