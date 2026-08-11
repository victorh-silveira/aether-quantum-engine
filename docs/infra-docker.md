# Infraestrutura Docker

Stack local **hibrida**: motor no host (Conda/WSL), persistencia e inferencia em containers. SSOT operacional deste doc; atalho em [`infra/docker/README.md`](../infra/docker/README.md).

## Servicos

| Servico | Porta (localhost) | Profile | Limite tipico | Uso |
|---------|-------------------|---------|---------------|-----|
| Redis | `127.0.0.1:6379` | `core` | 256m | Estado, risco, `settlement:queue:priority` |
| TimescaleDB | `127.0.0.1:5432` | `core` | 1g | Ticks + OHLC macro **3600 s** / micro **120 s** (M2) |
| MinIO | `127.0.0.1:9000` / `9001` | `core`, `gpu`, `cpu` | 512m | Checkpoints / TorchScript |
| Triton (`aether-triton`) | `127.0.0.1:8000` / `8001` | `gpu` ou `cpu` | — | Inferencia TorchScript HTTP+gRPC |
| Meta (`aether-meta-classifier`) | `127.0.0.1:8005` | `ml` | 512m | LGBMRegressor **43D**; `/v2/predict_meta` + `/v1/learn` online a cada settle (`META_RETRAIN_MIN_N` **2**); buffer `meta_learn_buffer.pkl` |
| Loss (`aether-loss-classifier`) | `127.0.0.1:8006` | `ml` | 512m | LGBMClassifier **24D**; buffer `learn_buffer.pkl` no volume; `/learn` + retrain **a cada trade** (WIN+LOSS no buffer; `LOSS_RETRAIN_MIN_N` **1**); saida bootstrap `LOSS_BOOTSTRAP_EXIT_N` **16** (floor efetivo ≥**8**); soft Kelly floor **0.65** / hard FLIP **0.90** |

Hardening: `restart: unless-stopped`, log rotate 10m×3, binds em **127.0.0.1**, `no-new-privileges` (onde aplicavel).

Logs de um servico: `make docker-logs DOCKER_SERVICE=<alias>`. Aliases Make → compose:

| Alias | Compose service |
|-------|-----------------|
| `redis`, `aether-redis` | `redis` |
| `ts`, `timescale`, `timescaledb`, `aether-timescaledb` | `timescaledb` |
| `minio`, `aether-minio` | `minio` |
| `triton`, `aether-triton` | `aether-triton` |
| `meta`, `meta-classifier`, `aether-meta-classifier` | `aether-meta-classifier` |
| `loss`, `loss-classifier`, `aether-loss-classifier` | `aether-loss-classifier` |

## Profiles e Make

| Target | Profiles | GPU overlay | Quando usar |
|--------|----------|-------------|-------------|
| `make docker-up` | `core,gpu,ml` (padrao) | sim (`DOCKER_GPU=1`) | Stack completa com NVIDIA |
| `make docker-up-cpu` | `core,cpu,ml` | nao | Triton sem NVIDIA (WSL CPU) |
| `make docker-up-core` | `core` | nao | So Redis/TS/MinIO (Triton off nos settings) |

**Exclusao mutua:** nao misturar `docker-up` (GPU) e `docker-up-cpu` na mesma porta 8000/8001. Overlay: [`docker-compose.gpu.yml`](../infra/docker/docker-compose.gpu.yml).

Pipeline `docker-up`: `host-prereq` → `triton-prereq` → compose up → wait healthy → timescale-lifecycle → hydrate (R_10 120/3600) → smoke.

Rebuild meta+loss: `make docker-rebuild` (sanitiza run host com `--keep-meta-bundle`, limpa `loss-models/` + seed `loss_bootstrap_synth`). Reset destrutivo: `make docker-reset` (sanitiza + `down --volumes`). Sanitizacao total (inclui `meta_lgbm.pkl` e `data/dl/*.pth`): `make sanitize-run` ou etapa 0 de `launch-train.bat`. Smoke: processo meta pode subir sem `.pkl` (aviso); modelo so apos `launch-train`.

## GPU e Triton

Imagem `nvcr.io/nvidia/tritonserver:24.10-py3`, repo bind `infra/docker/triton-models`. Flags: `--strict-readiness=false`, `--exit-on-error=false`. Health: `/v2/health/live`.

Layout de modelo deve ser `model.<backend>` (ex.: TorchScript); pasta `R_10` sem backend gera `Invalid model name` no log Triton — o motor hibrido usa CUDA local quando Triton nao carrega o simbolo.

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
| `AETHER_LOSS_CLASSIFIER_HTTP` | `http://localhost:8006` |
| `AETHER_DOCKER_HEALTH_TIMEOUT` | `300` |
| `DOCKER_PROFILES` / `COMPOSE_PROFILES` | `core,gpu,ml` |
| `DOCKER_GPU` | `1` (use `0` com `docker-up-cpu`) |

Settings app: `infra.redis.url`, `infra.timescale.dsn`, `infra.minio`, `infra.triton`, `infra.meta_classifier`, `infra.loss_classifier` — sempre **localhost** no hibrido.

## Redis / Timescale / MinIO

- Redis AOF `appendfsync everysec` (`redis.conf`); health com `start_period`
- Timescale: init `003_*.sql` + lifecycle `004_*.sql`; hydrate sintetico R_10 se micro&lt;360 ou macro&lt;80 (**nao** usar hydrate como unico historico para treino meta — preferir Deriv / `--source auto`)
- MinIO: bucket `dl-models`; health live + `start_period`
- Loss-classifier: volume `loss-models/`; bootstrap opcional `python -m scripts.operations.train_loss_classifier`
- **Reset operacional:** `make docker-rebuild` ou `make docker-reset` — limpa pkls, gera/seed `loss_bootstrap_synth` (`class_weight=balanced`, `min_child_samples=15`). Seed devolve **p_loss real** via `predict_proba` com `veto_ready=true` se `n_train>=ready_n` (**24**); sem `COLD_START` / sem `p_loss=0.50` neutro. Saida bootstrap exige buffer ≥**16** (`LOSS_BOOTSTRAP_EXIT_N`) com ≥1 WIN e ≥1 LOSS (floor efetivo ≥**8**); fit live colapsado e rejeitado (`collapsed_reject`). LEARN loga `retrain_skipped_reason`. Floor FLIP **0.90**. Apos rebuild do container, esperar `auto=1` em ~8–16 settles mistos.

## Relacao com o motor

Com `infra.enabled: true`, startup valida Redis/Timescale/MinIO (fail-fast). Mensagem operacional: `make docker-up-core|docker-up|docker-up-cpu`. Detalhe de software: [`arquitetura.md`](arquitetura.md). Skill: `aether-infra-stack`.
