# Infraestrutura Docker

Stack local **hibrida**: motor no host (Conda/WSL), persistencia e inferencia em containers. SSOT operacional deste doc; atalho em [`infra/docker/README.md`](../infra/docker/README.md).

## Servicos

| Servico | Porta (localhost) | Profile | Limite tipico | Uso |
|---------|-------------------|---------|---------------|-----|
| Redis | `127.0.0.1:6379` | `core` | 256m | Estado, risco, `settlement:queue:priority` |
| TimescaleDB | `127.0.0.1:5432` | `core` | 1g | Ticks + OHLC macro **600 s** / micro **120 s** |
| MinIO | `127.0.0.1:9000` / `9001` | `core`, `gpu`, `cpu` | 512m | Checkpoints / TorchScript |
| Triton (`aether-triton`) | `127.0.0.1:8000` / `8001` | `gpu` ou `cpu` | — | Inferencia TorchScript HTTP+gRPC |
| Meta (`aether-meta-classifier`) | `127.0.0.1:8005` | `ml` | 512m | LightGBM HTTP **43D** |

Hardening: `restart: unless-stopped`, log rotate 10m×3, binds em **127.0.0.1**, `no-new-privileges` (onde aplicavel).

## Profiles e Make

| Target | Profiles | GPU overlay | Quando usar |
|--------|----------|-------------|-------------|
| `make docker-up` | `core,gpu,ml` (padrao) | sim (`DOCKER_GPU=1`) | Stack completa com NVIDIA |
| `make docker-up-cpu` | `core,cpu,ml` | nao | Triton sem NVIDIA (WSL CPU) |
| `make docker-up-core` | `core` | nao | So Redis/TS/MinIO (Triton off nos settings) |

**Exclusao mutua:** nao misturar `docker-up` (GPU) e `docker-up-cpu` na mesma porta 8000/8001. Overlay: [`docker-compose.gpu.yml`](../infra/docker/docker-compose.gpu.yml).

Pipeline `docker-up`: `host-prereq` → `triton-prereq` → compose up → wait healthy → timescale-lifecycle → hydrate (R_10 120/600) → smoke.

Rebuild meta: `make docker-rebuild`. Smoke: `make docker-smoke` (falha se profile exige servico parado; meta exige JSON `ready`).

## GPU e Triton

Imagem `nvcr.io/nvidia/tritonserver:24.10-py3`, repo bind `infra/docker/triton-models`. Flags: `--strict-readiness=false`, `--exit-on-error=false`. Health: `/v2/health/live`.

Nos settings atuais o app pode ter `infra.triton.enabled: false`; a stack Docker permanece disponivel para fail-closed (`require_for_execution: true`).

Fluxo no motor: sync MinIO → `triton-models` → load explicito → sanity estressado → `TritonGrpcClient` em `localhost:8001`.

## Meta-regressor LightGBM

Porta host **8005**. Artefatos em `infra/docker/meta-models/` (`.pkl` **nao** versionado). Profile `ml` so fica healthy apos `train_meta_*`.

| Endpoint | Uso |
|----------|-----|
| `GET /health` | Exige `ready: true` |
| `POST /v2/predict_meta` | Vetor **43D** → `predicted_payoff_edge` |

Imagem: Python 3.13-slim, user nao-root `aether`.

## Variaveis (`.env`)

| Variavel | Padrao |
|----------|--------|
| `AETHER_TRITON_HTTP` / `AETHER_TRITON_GRPC` | `localhost:8000` / `localhost:8001` |
| `AETHER_META_CLASSIFIER_HTTP` | `http://localhost:8005` |
| `AETHER_DOCKER_HEALTH_TIMEOUT` | `300` |
| `DOCKER_PROFILES` / `COMPOSE_PROFILES` | `core,gpu,ml` |
| `DOCKER_GPU` | `1` (use `0` com `docker-up-cpu`) |

Settings app: `infra.redis.url`, `infra.timescale.dsn`, `infra.minio`, `infra.triton`, `infra.meta_classifier` — sempre **localhost** no hibrido.

## Redis / Timescale / MinIO

- Redis AOF `appendfsync everysec` (`redis.conf`)
- Timescale: init `003_*.sql` + lifecycle `004_*.sql`; hydrate sintetico R_10 se micro&lt;360 ou macro&lt;80
- MinIO: bucket `dl-models`

## Relacao com o motor

Com `infra.enabled: true`, startup valida Redis/Timescale/MinIO (fail-fast). Mensagem operacional: `make docker-up-core|docker-up|docker-up-cpu`. Detalhe de software: [`arquitetura.md`](arquitetura.md). Skill: `aether-infra-stack`.
