# Infraestrutura Docker

Stack local para o modo híbrido: motor no host (Conda/WSL), persistência e inferência em containers.

## Serviços

| Serviço | Porta | Uso |
|---------|-------|-----|
| Redis | 6379 | Estado, risco, assinaturas, starvation counter, fila `settlement:queue:priority` |
| TimescaleDB | 5432 | Ticks e barras OHLC (macro **600 s** + micro **120 s**; prefixos de assinatura legado `m15`/`m5`) |
| MinIO | 9000 / 9001 | Checkpoints Deep Learning / TorchScript |
| Triton (`aether-triton`) | 8000 / 8001 | Inferência GPU TorchScript via gRPC |
| Meta-regressor (`aether-meta-classifier`) | **8005** | LightGBM HTTP; vetor **43D** (fonte de verdade no app); `POST /v2/predict_meta` |

Subir tudo:

```bash
make docker-up
```

Pipeline: `host-prereq` → `triton-prereq` → `compose up` (profiles `DOCKER_PROFILES`, padrão `core,gpu,ml`) → wait healthy → `timescale-lifecycle` → `docker-hydrate` (seed OHLC de `R_10`) → `docker-smoke`.

| Profile | Serviços | Comando |
|---------|----------|---------|
| `core` | redis, timescaledb, minio | `make docker-up-core` |
| `gpu` | aether-triton (+ minio) | incluso em `docker-up` |
| `ml` | aether-meta-classifier | incluso em `docker-up` |

Rebuild do meta: `make docker-rebuild`. Smoke isolado: `make docker-smoke`.

## GPU e Triton

O serviço `aether-triton` usa `nvcr.io/nvidia/tritonserver:24.10-py3` com repositório em `infra/docker/triton-models` (bind mount). Requer **NVIDIA Container Toolkit** no WSL2. Compose declara `gpus: all` e reserva NVIDIA em `deploy.resources`.

### Fluxo de inferência

1. **Bootstrap**: `sync_all_symbols_to_triton` copia `latest_ts.pt` → `{symbol}/1/model.pt` + `config.pbtxt` com `fsync` antes do rename.
2. **Load-over-load**: `wait_triton_models_stable` dispara `POST /v2/repository/models/{name}/load` sequencial (MODE_EXPLICIT) apenas para modelos com artefato novo ou ainda nao ready — **nunca** `/unload`; aguarda ready entre simbolos.
3. **Sanity estressado**: `verify_triton_stressed_inference_async` (RSI/CMO/vol extremos); fail-fast se NaN/Inf ou prob fora de `[0, 1]`.
4. **Produção**: `TritonGrpcClient` mantém canal `grpc.aio.insecure_channel` persistente, inferências paralelas (`asyncio.gather`) e timeout configurável (`infra.triton.infer_timeout_seconds`, padrão **0,50 s**).
5. **Fail-closed**: com `infra.triton.require_for_execution: true`, timeout não cai para TorchScript local em produção.
6. **Loop-aware**: `get_triton_grpc_client` recria o singleton se o event loop asyncio mudou (treinos em thread / `asyncio.run`).

### Healthcheck Triton

`--strict-readiness=false` e `--exit-on-error=false` toleram repositório parcial antes do treino. Healthcheck via `python3` + `urllib` em `/v2/health/live` (HTTP 8000).

## Meta-regressor LightGBM

Serviço FastAPI na porta host **8005**. Artefatos em `infra/docker/meta-models/` (bind mount).

| Endpoint | Uso |
|----------|-----|
| `GET /health` | Healthcheck Docker |
| `POST /v2/predict_meta` | Regressão tabular; entrada: probabilidade TCN + vetor meta; saída: `predicted_payoff_edge` |

Dimensão canônica no app: **`META_FEATURE_DIM = 43`** (34 TCN + 4 micro-vol + 3 cross + 2 flow). Indicadores micro (RSI, shadow, momentum de spread) indexados em **120 s** no TimescaleDB. O artefato `.pkl` e o treino offline (`train_meta_*.py`) devem alinhar com essa dimensão e com a proporção multi-timeframe **1:5** (120:600).

Treino offline: `train_meta_classifier.py`, `train_meta_optuna.py`, `train_meta_vector.py` (Optuna maximiza Information Ratio; anti-leakage por proxy de retorno passado).

Variáveis no `.env`:

| Variável | Uso |
|----------|-----|
| `AETHER_META_CLASSIFIER_HTTP` | Endpoint (padrão `http://localhost:8005`) |
| `AETHER_TRITON_GRPC` | gRPC (padrão `localhost:8001`) |
| `AETHER_TRITON_HTTP` | HTTP (padrão `localhost:8000`) |
| `AETHER_DOCKER_HEALTH_TIMEOUT` | Timeout do wait healthy em segundos (padrão `300`) |
| `DOCKER_PROFILES` / `COMPOSE_PROFILES` | Profiles Compose (padrão Make: `core,gpu,ml`) |

Config em `settings.json`:

```json
"triton": {
  "enabled": true,
  "grpc_url": "localhost:8001",
  "http_url": "localhost:8000",
  "infer_timeout_seconds": 0.50,
  "require_for_execution": true,
  "model_repo_path": "infra/docker/triton-models"
},
"meta_classifier": {
  "enabled": true,
  "http_url": "http://localhost:8005",
  "timeout_seconds": 1.0
}
```

## Redis — pipeline atômico

`redis_state_pipeline.write_state_bundle` executa `MULTI/EXEC` com:

- `state:snapshot` (JSON completo)
- `state:risk` (hash: `consecutive_losses`, cooldowns, etc.)
- `state:pending_loss` (hash por símbolo)
- `session:current` (+ `start_balance` / `target_win`)
- `recovery:skip_counter`
- `state:risk:skipped_cycles_counter` (starvation do quality gate)
- `market_sig`
- `settlement:queue:priority` (ZSET; score = `contract_id`)

Gravado em `orchestrator_persistence.save_full_state` sob `StateManager._state_lock`. Redis local usa AOF `appendfsync everysec` (`infra/docker/redis.conf`).

## TimescaleDB e MinIO

- Timescale: writers de ticks/barras + worker de correlação.
- MinIO: source of truth remoto dos checkpoints; cache local em `data/dl/`.

## Relação com o motor

Com `infra.enabled: true`, o startup valida serviços (fail-fast), sincroniza Triton e executa sanity estressado **antes** do WebSocket Deriv. Detalhes de fluxo de software: [`arquitetura.md`](arquitetura.md).
