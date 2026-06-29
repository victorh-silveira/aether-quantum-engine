# Infraestrutura Docker do Aether

Stack local para Redis, TimescaleDB e MinIO. O motor (`run.py` / `train.py`) executa no host Conda/WSL e conecta via `localhost`.

## Subir servicos

```bash
cd infra/docker
cp .env.example .env
docker compose up -d
docker compose ps
```

## Portas

| Servico | Porta |
|---------|-------|
| Redis | 6379 |
| TimescaleDB | 5432 |
| MinIO API | 9000 |
| MinIO Console | 9001 |

## Pre-requisito do motor

Com `infra.enabled: true` em `config/settings.json`, o motor aborta o startup se algum servico estiver indisponivel (fail-fast).

Variaveis de ambiente no `.env` da raiz do repo:

- `AETHER_MINIO_ACCESS_KEY`
- `AETHER_MINIO_SECRET_KEY`

DSN padrao: `postgresql://aether:aether@localhost:5432/aether`
