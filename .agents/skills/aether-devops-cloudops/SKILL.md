---
name: aether-devops-cloudops
description: Endurecimento DevOps/CloudOps, Docker Compose, TimescaleDB, Redis, MinIO e limites de recursos.
---

# Endurecimento e Operação DevOps/CloudOps

## 1. Contexto e Pré-requisitos
- Leia `docs/engineering-devops-cloudops-senior.md` e `docs/infra-docker.md`.
- Certifique-se de ter Docker e Docker Compose instalados no host.

## 2. Procedimento Operacional Passo a Passo
1. **Auditoria do docker-compose.yml:**
   - Inspecione se todos os servicos possuem declaracoes de `mem_limit`, limites de CPU e diretivas `security_opt: ["no-new-privileges:true"]`.
   - Valide se as portas expostas estao atreladas estritamente a `127.0.0.1`.
2. **Inspeção de Saúde dos Containers:**
   - Valide os healthchecks declarados em cada servico (Redis, Timescale, MinIO, Grafana).
3. **Ciclo de Inicialização:**
   - Execute a rotina de subida com verificacao de saude:
     ```bash
     make docker-up
     make docker-status
     ```

## 3. Comandos de Verificação
```bash
make docker-status
bash infra/docker/sh/docker-smoke.sh
```

## 4. Critérios de Aceite
- Todos os containers em estado `healthy`.
- Zero portas abertas para interfaces externas.
