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

## Pre-requisito do host (WSL)

Antes de subir o Redis, aplique `vm.overcommit_memory=1` no kernel WSL:

```bash
make docker-up
```

O target executa `infra/docker/host-prereq.sh` automaticamente. No setup inicial do WSL:

```bash
make setup-wsl
```

Persistencia manual (opcional):

```bash
echo 'vm.overcommit_memory=1' | sudo tee /etc/sysctl.d/99-aether-redis.conf
sudo sysctl --system
```

Validacao Redis apos `docker compose up -d`:

```bash
docker exec -it aether-redis redis-cli CONFIG GET appendonly
docker exec -it aether-redis redis-cli CONFIG GET appendfsync
```

## Redis AOF

O servico usa `redis.conf` com `appendonly yes` e `appendfsync everysec` (RDB desabilitado via `save ""`).

## Pre-requisito do motor

Com `infra.enabled: true` em `config/settings.json`, o motor aborta o startup se algum servico estiver indisponivel (fail-fast).

Variaveis de ambiente no `.env` da raiz do repo:

- `AETHER_MINIO_ACCESS_KEY`
- `AETHER_MINIO_SECRET_KEY`

DSN padrao: `postgresql://aether:aether@localhost:5432/aether`
