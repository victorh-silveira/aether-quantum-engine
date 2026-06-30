# Infraestrutura Docker

Stack local para o modo hibrido: motor no host, persistencia em containers.

## Servicos

| Servico | Porta | Uso |
|---------|-------|-----|
| Redis | 6379 | Estado, risco, assinaturas de vela |
| TimescaleDB | 5432 | Ticks e barras OHLC |
| MinIO | 9000 / 9001 | Checkpoints Deep Learning |
| Triton (aether-triton) | 8000 / 8001 | Inferencia GPU TorchScript via gRPC |

## GPU e Triton

O servico `aether-triton` usa `nvcr.io/nvidia/tritonserver` com repositorio em `infra/docker/triton-models` (bind mount). Requer **NVIDIA Container Toolkit** no WSL2 para expor a GPU (ex.: RTX 4060).

O motor sincroniza `latest_ts.pt` do MinIO para o layout Triton (`{symbol}/1/model.pt` + `config.pbtxt`) no bootstrap e envia inferencia gRPC assincrona quando `infra.triton.enabled` estiver ativo.

Variaveis no `.env`:

| Variavel | Uso |
|----------|-----|
| `AETHER_TRITON_GRPC` | Endpoint gRPC (padrao `localhost:8001`) |
| `AETHER_TRITON_HTTP` | Health HTTP (padrao `localhost:8000`) |

## Subir

```bash
make docker-up
```

Ou manualmente (na raiz do repo):

```bash
cp .env.example .env
docker compose -f infra/docker/docker-compose.yml --project-directory infra/docker --env-file .env up -d
docker compose -f infra/docker/docker-compose.yml --project-directory infra/docker ps
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
  "minio": { "endpoint": "localhost:9000", "bucket": "dl-models", "secure": false },
  "triton": { "enabled": true, "grpc_url": "localhost:8001", "http_url": "localhost:8000" }
}
```

Variaveis no `.env` da raiz (unico arquivo para motor e Docker):

| Variavel | Uso |
|----------|-----|
| `AETHER_DERIV_PAT`, `AETHER_DERIV_APP_ID`, `AETHER_DERIV_ACCOUNT_ID` | API Deriv |
| `AETHER_PG_USER`, `AETHER_PG_PASSWORD`, `AETHER_PG_DB` | TimescaleDB |
| `AETHER_MINIO_ACCESS_KEY`, `AETHER_MINIO_SECRET_KEY` | MinIO |

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
