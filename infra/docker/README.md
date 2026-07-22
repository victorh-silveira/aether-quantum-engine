# Infraestrutura Docker do Aether

Stack local para Redis, TimescaleDB, MinIO, **Triton Inference Server** e **meta-regressor LightGBM**. O motor (`run.py` / `train.py`) executa no host Conda/WSL e conecta via `localhost`.

## Subir servicos

```bash
make docker-up
```

Pipeline do `make docker-up`:

| Etapa | Script / target | Funcao |
|-------|-----------------|--------|
| 1 | `host-prereq.sh` | `vm.overcommit_memory=1` no WSL |
| 2 | `triton-prereq.sh` | Remove `model.pt` invalidos; layout vazio ate `make train` |
| 3 | `docker compose up -d` | Sobe containers dos profiles ativos (`DOCKER_PROFILES`, padrao `core,gpu,ml`) |
| 4 | `docker-wait-healthy.sh` | Aguarda healthchecks (`healthy` estrito; timeout 300 s) |
| 5 | `timescale-lifecycle` | Compressao/retencao idempotente |
| 6 | `docker-hydrate` | Injeta OHLC macro/micro sintetico de `R_10` se `ohlc_bars` insuficiente |
| 7 | `docker-smoke` | Valida Redis / Timescale / MinIO / Triton / Meta |

Profiles Compose:

| Profile | Servicos | Target |
|---------|----------|--------|
| `core` | redis, timescaledb, minio | `make docker-up-core` |
| `gpu` | aether-triton (+ minio) | incluso em `docker-up` |
| `ml` | aether-meta-classifier | incluso em `docker-up` |

Rebuild da imagem do meta:

```bash
make docker-rebuild
```

Manual (raiz do repositorio):

```bash
cp .env.example .env
export COMPOSE_PROFILES=core,gpu,ml
docker compose -f infra/docker/docker-compose.yml --project-directory infra/docker --env-file .env up -d
docker compose -f infra/docker/docker-compose.yml --project-directory infra/docker ps
```

## Portas

| Servico | Porta | Uso |
|---------|-------|-----|
| Redis | 6379 | Estado, risco, assinaturas de vela, fila `settlement:queue:priority` |
| TimescaleDB | 5432 | Ticks e barras OHLC (macro **600 s** + micro **120 s**) |
| MinIO API | 9000 | Checkpoints Deep Learning |
| MinIO Console | 9001 | Console web |
| Triton HTTP | 8000 | Health live, metadata, reload |
| Triton gRPC | 8001 | Inferencia TorchScript em producao |
| Meta-classificador | **8005** | LightGBM HTTP; vetor **43D**; `POST /v2/predict_meta` |

## Triton e GPU

O servico `aether-triton` usa `nvcr.io/nvidia/tritonserver:24.10-py3` com repositorio em `infra/docker/triton-models` (bind mount). Requer **NVIDIA Container Toolkit** no WSL2.

Compose declara GPU via `gpus: all` e `deploy.resources.reservations.devices` (NVIDIA).

Flags de startup:

- `--strict-readiness=false` — tolera repo parcial antes do treino
- `--exit-on-error=false` — nao derruba o container por modelo ausente
- Healthcheck: `/v2/health/live` via Python urllib

Fluxo no motor:

1. `sync_all_symbols_to_triton` copia `latest_ts.pt` para `{symbol}/1/model.pt` (com fsync).
2. `wait_triton_models_stable` faz load explicito sequencial via HTTP na porta 8000.
3. `verify_triton_stressed_inference_async` valida inferencia sob tensores estressados.
4. `TritonGrpcClient` mantem canal gRPC persistente na porta 8001.

## Meta-classificador

Container `aether-meta-classifier`: Python 3.13-slim + FastAPI. Artefatos em `infra/docker/meta-models/` (`META_FEATURE_DIM=43`).

| Endpoint | Uso |
|----------|-----|
| `GET /health` | Healthcheck Docker (urllib nativo; exige `ready`) |
| `POST /v2/predict_meta` | Regressao tabular **43D** → `predicted_payoff_edge` |

Variaveis no `.env` da raiz:

| Variavel | Padrao |
|----------|--------|
| `AETHER_TRITON_GRPC` | `localhost:8001` |
| `AETHER_TRITON_HTTP` | `localhost:8000` |
| `AETHER_META_CLASSIFIER_HTTP` | `http://localhost:8005` |
| `AETHER_DOCKER_HEALTH_TIMEOUT` | `300` |
| `DOCKER_PROFILES` / `COMPOSE_PROFILES` | `core,gpu,ml` (via Make) |

## Pre-requisito do host (WSL)

```bash
make app-setup-wsl
make docker-up
```

Persistencia manual (opcional):

```bash
echo 'vm.overcommit_memory=1' | sudo tee /etc/sysctl.d/99-aether-redis.conf
sudo sysctl --system
```

## Redis AOF

`redis.conf`: `appendonly yes`, `appendfsync everysec`, RDB desabilitado.

## TimescaleDB

| Arquivo | Funcao |
|---------|--------|
| `003_init-timescale.sql` | Hypertables `ticks` / `ohlc_bars` |
| `004_timescale-lifecycle.sql` | Compressao 7 dias; retencao ticks 30 dias |

```bash
make timescale-lifecycle
```

Imagens pinadas: `timescale/timescaledb:2.28.2-pg16`, `minio/minio:RELEASE.2025-04-22T22-12-26Z`, `redis:7.4.2-alpine`.

## Pre-requisito do motor

Com `infra.enabled: true`, fail-fast valida Redis, Timescale, MinIO, sanity TorchScript, Triton estressado e health do meta-regressor (porta 8005).

Documentacao completa: [docs/infra-docker.md](../../docs/infra-docker.md) | [docs/structure.md](../../docs/structure.md)
