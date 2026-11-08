---
name: aether-architecture-senior
description: Audita e orienta mudancas na arquitetura senior Aether (host Python 3.13, DDD/hexagonal, asyncio, Polars SSOT, Docker).
---

# Auditoria e Evolução da Arquitetura Sênior

## 1. Contexto e Pré-requisitos
- Leia `AGENTS.md`, `docs/engineering-architecture-senior.md` e `docs/arquitetura.md`.
- Certifique-se de que o ambiente esta configurado com CPython 3.13.12 no host via Conda `deriv-api`.

## 2. Procedimento Operacional Passo a Passo
1. **Auditoria de Dependências de Camada:**
   - Inspecione `app/src/domain/` para assegurar que nenhum modulo importe `infrastructure`, bibliotecas de I/O (`httpx`, `websockets`, `redis`, `asyncpg`) ou modulos do sistema operacional.
   - Verifique que `app/src/application/` orquestre os fluxos atraves das interfaces definidas em `ports/outbound.py`.
2. **Inspeção de Limites de Linhas:**
   - Rode a verificacao estrutural para garantir que nenhum arquivo exceda 300 linhas:
     ```bash
     python app/scripts/operations/clean_workspace.py --area python --stage lint
     ```
3. **Auditoria de Manipulação Colunar:**
   - Assegure que qualquer agregacao tabular utilize exclusivamente Polars com Arrow zero-copy.
4. **Verificação de Isolamento:**
   - Garanta que servicos externos rodem estritamente em Docker Compose e que a comunicacao do host com sidecars ocorra via portas locais (`127.0.0.1`).

## 3. Comandos de Verificação
```bash
/mnt/c/Users/victor-silveira/anaconda3/envs/deriv-api/python.exe -m pytest app/tests/unit/domain/ -v
/mnt/c/Users/victor-silveira/anaconda3/envs/deriv-api/python.exe -m pytest app/tests/unit/application/test_architecture*.py -v
```

## 4. Critérios de Aceite (Definition of Done)
- Pureza estrita de `app/src/domain/` preservada com zero violacoes.
- Nenhum arquivo em `app/src/` acima de 300 linhas.
- Cobertura de testes unitarios em 100%.
