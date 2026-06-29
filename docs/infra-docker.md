# Infraestrutura Docker

Stack local para o modo hibrido: motor no host, persistencia em containers.

## Servicos

| Servico | Porta | Uso |
|---------|-------|-----|
| Redis | 6379 | Estado, risco, assinaturas de vela |
| TimescaleDB | 5432 | Ticks e barras OHLC |
| MinIO | 9000 / 9001 | Checkpoints Deep Learning |

## Subir

```bash
make docker-up
```

Ou manualmente:

```bash
cd infra/docker
cp .env.example .env
docker compose up -d
docker compose ps
```

## Host WSL e Redis

`make docker-up` executa `infra/docker/host-prereq.sh`, que tenta definir `vm.overcommit_memory=1` no kernel WSL (evita warning do Redis). O `make setup-wsl` tambem invoca esse script.

Persistencia Redis: AOF com `appendfsync everysec` via `infra/docker/redis.conf` (RDB desabilitado).

Validacao:

```bash
docker exec -it aether-redis redis-cli CONFIG GET appendonly
docker exec -it aether-redis redis-cli CONFIG GET appendfsync
```

Persistencia manual do sysctl (opcional):

```bash
echo 'vm.overcommit_memory=1' | sudo tee /etc/sysctl.d/99-aether-redis.conf
sudo sysctl --system
```

## Configuracao do motor

Bloco `infra` em `config/settings.json`:

```json
"infra": {
  "enabled": true,
  "fail_fast": true,
  "redis": { "url": "redis://localhost:6379/0", "key_prefix": "aether" },
  "timescale": { "dsn": "postgresql://aether:aether@localhost:5432/aether" },
  "minio": { "endpoint": "localhost:9000", "bucket": "dl-models", "secure": false }
}
```

Variaveis no `.env` da raiz:

- `AETHER_MINIO_ACCESS_KEY`
- `AETHER_MINIO_SECRET_KEY`

## Fail-fast

Antes do WebSocket Deriv, o motor executa:

- Redis `PING`
- TimescaleDB `SELECT 1`
- MinIO `HeadBucket`

Se algum falhar e `fail_fast` estiver ativo, o processo encerra com mensagem indicando `docker compose up -d`.

## Schema Timescale

Hypertables `ticks` e `ohlc_bars` criadas por `infra/docker/init-timescale.sql` no primeiro boot do container.

## Testes de integracao

Testes com marker `@pytest.mark.docker` em `app/tests/integration/` nao rodam no pre-commit.
