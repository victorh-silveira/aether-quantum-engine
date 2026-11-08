---
name: aether-polars-arrow
description: Manipulacao colunar de alta performance com Polars/Arrow zero-copy e banimento de pandas.
---

# Manipulação de Dados com Polars e Arrow

## 1. Contexto e Pré-requisitos
- Leia `docs/engineering-python-deps.md` e `.agents/rules/aether-python-deps.mdc`.

## 2. Procedimento Operacional Passo a Passo
1. **Auditoria de Pipelines de Dados:**
   - Garanta que series temporais e calculos de indicadores utilizem `polars.LazyFrame` para otimizacao de grafos de execucao.
2. **Conversão Zero-Copy:**
   - Quando for necessario interoperar com tensores ou NumPy, utilize o protocolo PyCapsule ou `to_arrow()` / `to_numpy(zero_copy_only=True)`.
3. **Controle de Paralelismo:**
   - Configure `POLARS_MAX_THREADS` de forma que operacoes de agregação nao saturem os nucleos da CPU reservados para o event loop.

## 3. Comandos de Verificação
```bash
/mnt/c/Users/victor-silveira/anaconda3/envs/deriv-api/python.exe -m pytest app/tests/unit/repo/test_python_deps_policy.py -v
```

## 4. Critérios de Aceite
- Zero ocorrencias de pandas no repositorio.
- Transformacoes de dados eficientes com Polars.
