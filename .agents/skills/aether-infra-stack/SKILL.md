---
name: aether-infra-stack
description: Inicializacao, monitoramento e saneamento da stack Docker do Aether (Redis, Timescale, MinIO).
---

# Gerenciamento da Stack de Infraestrutura Docker

## 1. Contexto e Pré-requisitos
- Leia `docs/infra-docker.md` e `.agents/rules/aether-infra.mdc`.

## 2. Procedimento Operacional Passo a Passo
1. **Subida da Infraestrutura:**
   - Execute o target oficial do Makefile:
     ```bash
     make docker-up
     ```
2. **Inspeção de Logs e Saúde:**
   - Verifique se todos os servicos reportam status `healthy`:
     ```bash
     make docker-status
     docker compose -f infra/docker/docker-compose.yml ps
     ```
3. **Parada Segura:**
   - Quando necessario encerrar:
     ```bash
     make docker-down
     ```

## 3. Comandos de Verificação
```bash
bash infra/docker/sh/docker-smoke.sh
```

## 4. Critérios de Aceite
- Todos os servicos ativos e respondendo em `127.0.0.1`.
