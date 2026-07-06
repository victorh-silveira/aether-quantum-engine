# Infraestrutura Docker

Stack local para o modo híbrido: motor no host, persistência em containers.

## Serviços

| Serviço | Porta | Uso |
|---------|-------|-----|
| Redis | 6379 | Estado, risco, assinaturas de vela, `recovery:skip_counter` |
| TimescaleDB | 5432 | Ticks e barras OHLC |
| MinIO | 9000 / 9001 | Checkpoints Deep Learning |
| Triton (`aether-triton`) | 8000 / 8001 | Inferência GPU TorchScript via gRPC |
| Meta-regressor (`aether-meta-classifier`) | **8005** | LightGBM HTTP; vetor **39D**; `POST /v2/predict_meta` → `predicted_payoff_edge` |

## GPU e Triton

O serviço `aether-triton` usa `nvcr.io/nvidia/tritonserver` com repositório em `infra/docker/triton-models` (bind mount). Requer **NVIDIA Container Toolkit** no WSL2 para expor a GPU.

### Fluxo de inferência

1. **Bootstrap**: `sync_all_symbols_to_triton` copia `latest_ts.pt` para `{symbol}/1/model.pt` + `config.pbtxt`.
2. **Reload**: `reload_triton_repository` via HTTP após sync.
3. **Sanity estressado**: `verify_triton_stressed_inference_async` envia tensores com RSI=0.99, CMO=1.0, vol_ratio=1.80; fail-fast se NaN/Inf ou prob fora de `[0, 1]`.
4. **Produção**: `TritonGrpcClient` mantém canal `grpc.aio.insecure_channel` persistente, dispara inferências de `RDBEAR` e `RDBULL` em paralelo (`asyncio.gather`) e aplica **timeout de 2,0 s** por requisição. Em timeout, `dl_predict_triton` faz fallback para TorchScript local (log `TRITON_TIMEOUT_FALLBACK`).

### Healthcheck Triton

O healthcheck do container `aether-triton` usa `python3` + `urllib` contra `/v2/health/ready` (porta HTTP 8000), sem dependência de `curl` na imagem base.

## Meta-regressor LightGBM

O serviço `aether-meta-classifier` expõe FastAPI na porta host **8005** com `FEATURE_DIM=39`. Artefatos em `infra/docker/meta-models/` (bind mount).

| Endpoint | Uso |
|----------|-----|
| `GET /health` | Healthcheck Docker via `urllib` nativo |
| `POST /v2/predict_meta` | Regressão tabular; entrada: probabilidade TCN + vetor 39D; saída: `predicted_payoff_edge` |

Treino offline: `train_meta_classifier.py`, `train_meta_optuna.py` e `train_meta_vector.py` (Optuna minimiza MAE; `LGBMRegressor` huber com `feature_name=columns`; alvo `Y = PnL_Real / Stake`; sumário com `train_mae` e `target_variance`).

Variáveis no `.env`:

| Variável | Uso |
|----------|-----|
| `AETHER_META_CLASSIFIER_URL` | Endpoint HTTP (padrão `http://localhost:8005`) |

| Variável | Uso |
|----------|-----|
| `AETHER_TRITON_GRPC` | Endpoint gRPC (padrão `localhost:8001`) |
| `AETHER_TRITON_HTTP` | Health/metadata HTTP (padrão `localhost:8000`) |

Config em `settings.json`:

```json
"triton": {
  "enabled": true,
  "grpc_url": "localhost:8001",
  "http_url": "localhost:8000",
  "model_repo_path": "infra/docker/triton-models"
}
```

## Redis — pipeline atômico

`redis_state_pipeline.write_state_bundle` executa `MULTI/EXEC` com:

- `state:snapshot` (JSON completo)
- `state:risk` (hash: `consecutive_losses`, cooldowns, etc.)
- `state:pending_loss` (hash por símbolo)
- `session:current` (hash: banca corrente, trades, meta win)
- `session:current:start_balance` (string — banca inicial da sessão ativa)
- `session:current:target_win` (string — meta de lucro da sessão)
- `recovery:skip_counter` (decaimento Hurst em recovery)
- `market_sig` (assinatura OHLC)

As chaves `session:current:*` são gravadas no bootstrap da sessão e removidas no `graceful_shutdown` (ou no fast-path de stop win, **antes** do shutdown, via `clear_current_session_redis_keys`). Cada restart do processo inicia uma sessão independente.

Gravado em `save_full_state` após cada settlement, sem bloquear a thread principal com múltiplos round-trips.

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

`make docker-up` executa `infra/docker/host-prereq.sh`, que tenta definir `vm.overcommit_memory=1` no kernel WSL (evita warning do Redis). O `make setup-wsl` também invoca esse script.

Persistência Redis: AOF com `appendfsync everysec` via `infra/docker/redis.conf` (RDB desabilitado).

TimescaleDB: `checkpoint_completion_target=0.95` e `max_wal_size=2GB` via flags do `postgres` no `docker-compose.yml`. Novos volumes aplicam `002_aether-io-tune.sql` no init.

Validação:

```bash
docker exec -it aether-redis redis-cli CONFIG GET appendonly
docker exec -it aether-redis redis-cli CONFIG GET appendfsync
```

## Configuração do motor

Bloco `infra` em `config/settings.json`:

```json
"infra": {
  "enabled": true,
  "fail_fast": true,
  "redis": { "url": "redis://localhost:6379/0", "key_prefix": "aether" },
  "timescale": { "dsn": "postgresql://aether:aether@localhost:5432/aether" },
  "minio": { "endpoint": "localhost:9000", "bucket": "dl-models", "secure": false },
  "triton": { "enabled": true, "grpc_url": "localhost:8001", "http_url": "localhost:8000" },
  "meta_classifier": { "enabled": true, "url": "http://localhost:8005" }
}
```

Variáveis no `.env` da raiz (único arquivo para motor e Docker):

| Variável | Uso |
|----------|-----|
| `AETHER_DERIV_PAT`, `AETHER_DERIV_APP_ID`, `AETHER_DERIV_ACCOUNT_ID` | API Deriv |
| `AETHER_PG_USER`, `AETHER_PG_PASSWORD`, `AETHER_PG_DB` | TimescaleDB |
| `AETHER_MINIO_ACCESS_KEY`, `AETHER_MINIO_SECRET_KEY` | MinIO |

## Fail-fast

Antes do WebSocket Deriv, o motor executa:

- Redis `PING`
- TimescaleDB `SELECT 1`
- MinIO `HeadBucket`
- Sanity TorchScript local (probes + regime estressado)
- Com Triton: schema HTTP + inferência estressada concorrente
- Com meta-regressor: `GET /health` na porta 8005

Se algum falhar e `fail_fast` estiver ativo, o processo encerra com mensagem indicando `docker compose up -d`.

## Schema Timescale

Hypertables `ticks` e `ohlc_bars` criadas por `infra/docker/003_init-timescale.sql` no primeiro boot do container.

Politicas de ciclo de vida (idempotentes) em `infra/docker/004_timescale-lifecycle.sql`:

| Hypertable | Compressao | Retencao |
|------------|------------|----------|
| `ticks` | Chunks com mais de **7 dias** (`compress_segmentby = symbol`, `compress_orderby = time DESC`) | Drop automatico apos **30 dias** |
| `ohlc_bars` | Chunks com mais de **7 dias** (mesma segmentacao) | Sem retention (correlacao cross-symbol) |

Ordem de bootstrap em `docker-entrypoint-initdb.d/`: `002_aether-io-tune.sql` -> `003_init-timescale.sql` -> `004_timescale-lifecycle.sql` (prefixo numerico evita execucao prematura).

Se o primeiro boot falhou (volume parcialmente inicializado), recrie o volume antes de subir de novo:

```bash
docker compose -f infra/docker/docker-compose.yml --project-directory infra/docker down
docker volume rm docker_aether_ts_data
make docker-up
```

Volumes existentes recebem as mesmas politicas via `make timescale-lifecycle` (executado automaticamente em `make docker-up`).

Validacao:

```bash
docker exec -it aether-timescaledb psql -U aether -d aether -c \
  "SELECT hypertable_name, job_id, schedule_interval, config FROM timescaledb_information.jobs WHERE proc_name IN ('policy_compression','policy_retention');"
```

## Testes de integração

Testes com marker `@pytest.mark.docker` em `app/tests/integration/` não rodam no pre-commit.
