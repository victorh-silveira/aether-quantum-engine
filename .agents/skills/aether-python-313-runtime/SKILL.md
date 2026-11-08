---
name: aether-python-313-runtime
description: Auditoria de runtime CPython 3.13, GIL, otimizacoes de interpretador e memoria.
---

# Auditoria do Runtime CPython 3.13

## 1. Contexto e Pré-requisitos
- Leia `docs/engineering-python-313-runtime.md` e `.agents/rules/aether-python-313-runtime.mdc`.

## 2. Procedimento Operacional Passo a Passo
1. **Auditoria de Interpretador:**
   - Confirme a versao oficial: `python --version` deve reportar Python 3.13.12.
2. **Inspeção de Alocações no Hot-Path:**
   - Evite instanciacao repetitiva de objetos e dicionarios dentro do callback de ticks. Reutilize buffers estruturados.
3. **Uso de Slots e Imutabilidade:**
   - Classes criticas de mensagens de mercado devem utilizar `__slots__` para reducao de footprint de memoria e acesso veloz a atributos.

## 3. Comandos de Verificação
```bash
/mnt/c/Users/victor-silveira/anaconda3/envs/deriv-api/python.exe -m pytest app/tests/unit/infrastructure/test_runtime*.py -v
```

## 4. Critérios de Aceite
- Conformidade total com CPython 3.13.
