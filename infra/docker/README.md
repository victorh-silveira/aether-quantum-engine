# Infraestrutura Docker do Aether

Stack local para Redis, TimescaleDB, MinIO, **Triton Inference Server** e **meta-regressor LightGBM**. O motor (`run.py` / `train.py`) executa no host Conda/WSL e conecta via `localhost`.

## Subir servicos

```bash
make docker-up
```

Ou manualmente (a partir da raiz do repositorio):

```bash
cp .env.example .env
docker compose -f infra/docker/docker-compose.yml --project-directory infra/docker --env-file .env up -d
docker compose -f infra/docker/docker-compose.yml --project-directory infra/docker ps
```

## Portas

| Servico | Porta | Uso |
|---------|-------|-----|
| Redis | 6379 | Estado, risco, assinaturas de vela |
| TimescaleDB | 5432 | Ticks e barras OHLC |
| MinIO API | 9000 | Checkpoints Deep Learning |
| MinIO Console | 9001 | Console web |
| Triton HTTP | 8000 | Health, metadata, reload do repositório |
| Triton gRPC | 8001 | Inferência TorchScript em produção |
| Meta-classificador | **8005** | LightGBM HTTP; vetor 39D; `POST /v2/predict_meta` |

## Triton e GPU

O serviço `aether-triton` usa `nvcr.io/nvidia/tritonserver` com repositório em `infra/docker/triton-models` (bind mount). Requer **NVIDIA Container Toolkit** no WSL2.

Fluxo no motor:

1. `sync_all_symbols_to_triton` copia `latest_ts.pt` para `{symbol}/1/model.pt`.
2. `reload_triton_repository` via HTTP na porta 8000.
3. `verify_triton_stressed_inference_async` valida inferência sob tensores estressados.
4. `TritonGrpcClient` mantém canal gRPC persistente na porta 8001.

Healthcheck Triton: `python3` + `urllib` em `/v2/health/ready` (sem `curl`).

## Meta-classificador

O serviço `aether-meta-classifier` usa imagem Python 3.13-slim com FastAPI. Artefatos LightGBM em `infra/docker/meta-models/` (`FEATURE_DIM=39`).

| Endpoint | Uso |
|----------|-----|
| `GET /health` | Healthcheck Docker (urllib nativo) |
| `POST /v2/predict_meta` | Inferência tabular 39D |

Healthcheck: `python3` + `urllib` em `http://localhost:8005/health`.

Variáveis no `.env` da raiz:

| Variável | Padrão |
|----------|--------|
| `AETHER_TRITON_GRPC` | `localhost:8001` |
| `AETHER_TRITON_HTTP` | `localhost:8000` |
| `AETHER_META_CLASSIFIER_URL` | `http://localhost:8005` |

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

O motor grava estado via `orchestrator_persistence.save_full_state` → `redis_state_pipeline.write_state_bundle` (MULTI/EXEC atomico), incluindo `recovery:skip_counter` e chaves de sessao ativa (`session:current:start_balance`, `session:current:target_win`). A gravacao ocorre sob `StateManager._state_lock`; metricas locais adicionais em `data/session_state.json`.

## TimescaleDB: compressao e retencao

Arquivos SQL:

| Arquivo | Funcao |
|---------|--------|
| `003_init-timescale.sql` | Extension, hypertables `ticks`/`ohlc_bars`, indices |
| `004_timescale-lifecycle.sql` | Compressao columnar (7 dias) e retention de ticks (30 dias) |

Politicas assincronas nativas do TimescaleDB (`add_compression_policy`, `add_retention_policy`) com `if_not_exists => TRUE`.

Apos `make docker-up`, `make timescale-lifecycle` reaplica politicas em volumes ja existentes (idempotente).

```bash
make timescale-lifecycle
docker exec -it aether-timescaledb psql -U aether -d aether -c \
  "SELECT * FROM timescaledb_information.compression_settings;"
```

## Pre-requisito do motor

Com `infra.enabled: true` em `config/settings.json`, o motor aborta o startup se algum servico estiver indisponivel (fail-fast), incluindo sanity TorchScript local, inferência estressada no Triton quando `infra.triton.enabled` e health do meta-regressor quando `infra.meta_classifier.enabled`.

Todas as variaveis de ambiente ficam no `.env` da **raiz** do repositorio (Deriv, Postgres, MinIO e Triton). Copie de `.env.example`:

```bash
cp .env.example .env
```

Chaves usadas pelo Docker Compose:

- `AETHER_PG_USER`, `AETHER_PG_PASSWORD`, `AETHER_PG_DB`
- `AETHER_MINIO_ACCESS_KEY`, `AETHER_MINIO_SECRET_KEY`

Chaves usadas pelo motor Deriv:

- `AETHER_DERIV_PAT`, `AETHER_DERIV_APP_ID`, `AETHER_DERIV_ACCOUNT_ID` (opcional)

Documentação completa: [docs/infra-docker.md](../../docs/infra-docker.md).
