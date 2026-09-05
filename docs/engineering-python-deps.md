# Dependencias Python (SSOT sênior)

Pins: [`app/requirements.txt`](../app/requirements.txt), [`app/requirements-dev.txt`](../app/requirements-dev.txt), Docker `infra/docker/*/requirements.txt`. Runtime: [`engineering-python-313-runtime.md`](engineering-python-313-runtime.md). CloudOps: [`engineering-devops-cloudops-senior.md`](engineering-devops-cloudops-senior.md).

Rule: `aether-python-deps.mdc`. Skill: `aether-python-deps` (+ `aether-polars-arrow`, `aether-torch-cuda-infer`, `aether-redis-hiredis`, `aether-asyncpg-timescale`, `aether-deriv-connect`).

## Stack host (pins)

| Lib | Versao | Papel |
|-----|--------|--------|
| websockets | 16.0 | WSS Deriv |
| httpx | 0.28.1 | HTTP async (meta/loss) |
| python-dotenv | 1.2.2 | `.env` bootstrap |
| numpy | 2.4.6 | arrays / ML |
| polars | 1.23.0 | DataFrame SSOT |
| rich | 15.0.0 | terminal |
| redis[hiredis] | >=5.0 | estado + settlement |
| asyncpg | >=0.30 | Timescale |
| minio | >=7.2 | artefatos S3 |
| torch | 2.10.0 | TCN CUDA host |
| lightgbm | 4.6.0 | meta/loss tabular |
| optuna | 4.2.1 | HPO offline |
| scikit-learn | 1.6.1 | metricas |
| joblib | 1.4.2 | serializacao meta/loss |

Sidecars ML: FastAPI + uvicorn[standard] (uvloop/httptools) em `python:3.13-slim`.

Dev: ruff 0.15.13, pre-commit 4.6.0, interrogate 1.7.0, vulture 2.16, bandit 1.9.4, pip-audit 2.10.0, pytest 9.0.3 + asyncio/mock/timeout/xdist/cov.

---

## 1. Rede e streaming

### websockets 16.0

- Internals: sans-I/O + `websockets.speedups` (C/Cython); frames RFC 6455 desacoplados do transporte asyncio.
- **SSOT connect:** `max_size=4_194_304` (4 MiB), `ping_interval=20`, `ping_timeout=10` (WSL/NAT half-open). Defaults em `websocket_connect.apply_websocket_connect_defaults`.
- Heartbeat de app (`{"ping": 1}`) permanece; ping da lib cobre sockets mortos.
- **Armadilhas:** `PayloadTooBigError` se payload > `max_size`; `async for` sem consumo gera backpressure TCP; nao omitir `max_size`/`ping_*` em connects novos.

### httpx 0.28.1

- Pools HTTP/1.1 via httpcore; **singleton** por loop (`meta_classifier_pool` / `loss_classifier_pool` + `build_persistent_http_client` com `Limits`).
- **Proibido:** `AsyncClient()` por request / por sinal (renegocia TLS e esgota FDs).
- Timeouts padrao httpx sao estritos (~5s); sidecars ML calibram timeout no client.
- Bridge sync (`predict_*_via_config_sync`) sob event loop vivo recria client — anti-padrao residual; preferir caminho async.

### python-dotenv 1.2.2

- So no bootstrap; `load_dotenv()` default `override=False` (env do SO/Compose ganha).
- Tudo e `str` — parse/validacao em infra, nao no hot path de ticks.

---

## 2. Vetorizacao

### numpy 2.4.6

- ABI 2.x / DType API; wheels C devem ser binarios compatíveis.
- Preferir views (slice, `ravel`) a `.flatten()` / copias.
- Type promotion 2.x e mais estrita (evitar promover tudo a float64 por acidente).

### polars 1.23.0 (SSOT DataFrame)

- Arrow columnar; **pandas proibido** (`to_pandas`, dual-stack, 3a lib DF).
- Preferir `LazyFrame` (predicate/projection pushdown); codigo novo nao deve encadear eager materializando a cada passo.
- Polars libera o GIL (Rayon): queries pesadas → `asyncio.to_thread` / executor.
- Host: `POLARS_MAX_THREADS=2` via `setdefault` no bootstrap (`app/run.py`) para nao saturar o loop.

---

## 3. Estado, persistencia, artefatos

### redis[hiredis]

- `redis.asyncio` + parser C RESP; pipelines para RTT baixo.
- Settlement SSOT: ZSET `settlement:queue:priority` (nunca Streams sem mandato).
- `decode_responses=True` ok para estado texto; payloads binarios de baixa latencia preferem bytes.
- Pool: nao esgotar `max_connections` sob fan-out de tasks.

### asyncpg

- Prepared LRU por conexao; `copy_records_to_table` / batch para ticks.
- Direto ao Timescale (sem PgBouncer transaction mode).
- `CancelledError` descarta conexao contaminada — monitorar churn do pool; evitar `idle in transaction`.

### minio

- SDK sync/urllib3: **sempre** `await asyncio.to_thread(...)` ([`minio_model_store.py`](../app/src/infrastructure/storage/minio_model_store.py)).
- Validar integridade (ETag/hash) antes de `joblib`/torch load.

---

## 4. ML

### torch 2.10.0

- Inferencia: `torch.inference_mode()`; hot path `to_thread(eager_local_predict)`.
- Warm-up CUDA no bootstrap (WSL frio).
- `torch.compile` = opt-in offline; nao obrigar no DEMO live.

### lightgbm 4.6.0

- Sidecar: `n_jobs=2` + `OMP_NUM_THREADS=2` (Compose/Dockerfile).
- Host train: `n_jobs=1` tipico (nao disputar VRAM/CPU com inferencia).

### optuna 4.2.1

- HPO offline; storage paralelo: evitar SQLite no FS WSL (locks); preferir RDB/Redis.

### scikit-learn / joblib

- Fit de scalers so no train (anti lookahead).
- `joblib` pin direto; modelos do MinIO com hash antes de unpickle (pickle arbitrario).

---

## 5. Dev / QA

| Ferramenta | Papel |
|------------|--------|
| ruff | lint+format (E,F,B,I,UP,ASYNC) |
| pre-commit | gates isolados |
| interrogate / vulture | docstrings / morto (Protocols: cuidado falso positivo) |
| bandit / pip-audit | SAST / CVE |
| pytest-asyncio | `asyncio_mode`; escopo de loop alinhado a fixtures session |
| pytest-xdist | isolar Redis/TS por worker |
| pytest-timeout | hangs asyncio |
| pytest-cov | **CI / validacao**; nao no TDD local iterativo |

Nao pinar `coverage` se `pytest-cov` ja esta no requirements-dev.

---

## Principios de governanca

1. Declarar todo `import` first-party no requirements do ambiente certo.
2. Nao pinar so-transitivo de wrapper.
3. Uma lib por papel; DataFrame = **somente Polars**.
4. Apos bump numpy/torch/sklearn: `pip check` + smoke import no WSL.

| Camada | Libs | Padrao sênior |
|--------|------|----------------|
| Network | websockets, httpx | reuse; `max_size`/ping; sem client por request |
| Vetores | numpy, polars | zero-copy/views; Lazy; offload do loop |
| Estado | redis, asyncpg, minio | pipelines; COPY/batch; MinIO em thread |
| ML | torch, lightgbm, joblib | inference_mode; threads limitadas; hash artefato |
| QA | ruff, pytest-* | loop scope; xdist isolado; cov no CI |

## Anti-padroes

- `AsyncClient()` no hot path; omitir `max_size`/`ping_*` no WSS
- Polars eager em cadeia + query pesada no thread do loop
- MinIO sync na corrotina; LightGBM `n_jobs=-1` no sidecar
- pandas / `to_pandas`; Streams no settlement
- SQLite Optuna paralelo no WSL; joblib sem validar origem

## Onde editar

| Ambiente | Arquivo |
|----------|---------|
| Runtime / treino host | `app/requirements.txt` |
| Pre-commit / testes | `app/requirements-dev.txt` |
| Containers meta / loss | `infra/docker/*/requirements.txt` |
