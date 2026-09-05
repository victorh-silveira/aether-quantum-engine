# Infraestrutura Docker

Stack local **hibrida**: motor no host (Conda/WSL, Python 3.13, CUDA local) com inferencia TCN **eager** no processo do motor; persistencia e sidecars ML em containers. Doutrina CloudOps: [`engineering-devops-cloudops-senior.md`](engineering-devops-cloudops-senior.md). Arquitetura: [`engineering-architecture-senior.md`](engineering-architecture-senior.md). SSOT operacional deste doc; atalho em [`infra/docker/README.md`](../infra/docker/README.md).

## Servicos

| Servico | Porta (localhost) | Profile | Limite tipico | Uso |
|---------|-------------------|---------|---------------|-----|
| Redis | `127.0.0.1:6379` | `core` | 256m | Estado, risco, `settlement:queue:priority` (AOF everysec, `maxmemory`/`noeviction`) |
| TimescaleDB | `127.0.0.1:5432` | `core` | 1g | Ticks + OHLC macro **86400 s** (D1) / micro **300 s** (M5); chunk 1d; CRAG `candle_m5` analytics |
| MinIO | `127.0.0.1:9000` / `9001` | `core` | 512m | Checkpoints / TorchScript; bucket `dl-models`; `minio-init` + ILM `optuna/` ~7d |
| Meta (`aether-meta-classifier`) | `127.0.0.1:8005` | `ml` | 512m | LGBMRegressor **43D**; `/v2/predict_meta` + `/v1/learn` online a cada settle (`META_RETRAIN_MIN_N` **2**); buffer `meta_learn_buffer.pkl` |
| Loss (`aether-loss-classifier`) | `127.0.0.1:8006` | `ml` | 512m | LGBMClassifier **24D**; buffer `learn_buffer.pkl` no volume; `/learn` + retrain **a cada trade** (WIN+LOSS no buffer; `LOSS_RETRAIN_MIN_N` **1**); saida bootstrap `LOSS_BOOTSTRAP_EXIT_N` **8**; soft Kelly floor **0.65** / hard FLIP **0.90** |

Hardening: `restart: unless-stopped`, log rotate 10m×3, binds em **127.0.0.1**, `mem_swappiness: 0`, `no-new-privileges`, `depends_on: service_healthy` onde aplicavel, OMP*=2 nos ML.

Logs de um servico: `make docker-logs DOCKER_SERVICE=<alias>`. Aliases Make → compose:

| Alias | Compose service |
|-------|-----------------|
| `redis`, `aether-redis` | `redis` |
| `ts`, `timescale`, `timescaledb`, `aether-timescaledb` | `timescaledb` |
| `minio`, `aether-minio` | `minio` |
| `meta`, `meta-classifier`, `aether-meta-classifier` | `aether-meta-classifier` |
| `loss`, `loss-classifier`, `aether-loss-classifier` | `aether-loss-classifier` |

## Profiles e Make

| Target | Profiles | Quando usar |
|--------|----------|-------------|
| `make docker-up` | `core,ml` (padrao) | Stack completa: Redis/TS/MinIO + meta + loss |
| `make docker-up-core` | `core` | So Redis/TS/MinIO |

Pipeline `docker-up`: `host-prereq` → compose up → wait healthy → timescale-lifecycle → hydrate (1HZ75V micro/macro) → smoke.

Fluxo diario: `make docker-up` → `launch-train.bat` (sanitiza + treina TCN/meta) → `make docker-rebuild` (reconstroi imagens meta/loss e recarrega pkls **sem** apagar `data/dl`). Rebuild **nao** chama `sanitize_fresh_run`. Reset destrutivo: `make docker-reset` (sanitiza TCN/loss/estado, mantem `meta_lgbm.pkl`, `down --volumes`). Sanitizacao total (inclui `meta_lgbm.pkl` e `data/dl/*.pth`): `make sanitize-run` ou etapa 0 de `launch-train.bat`. Smoke: processo meta pode subir sem `.pkl` (aviso); modelo so apos `launch-train`.

## Inferencia TCN (host)

TCN roda no processo do motor (PyTorch eager / CUDA local) a partir de checkpoints em `data/dl/`. MinIO guarda artefactos; nao ha servidor de inferencia no compose. Sanity de TorchScript (quando aplicavel) e local ao host.

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
| `AETHER_META_CLASSIFIER_HTTP` | `http://localhost:8005` |
| `AETHER_LOSS_CLASSIFIER_HTTP` | `http://localhost:8006` |
| `AETHER_DOCKER_HEALTH_TIMEOUT` | `300` |
| `DOCKER_PROFILES` / `COMPOSE_PROFILES` | `core,ml` |

Settings app: `infra.redis.url`, `infra.timescale.dsn`, `infra.minio`, `infra.meta_classifier`, `infra.loss_classifier` — sempre **localhost** no hibrido.

## Redis / Timescale / MinIO

- Redis AOF `appendfsync everysec` (`redis.conf`); health com `start_period`
- Timescale: init `003_*.sql` + lifecycle `004_*.sql`; hydrate sintetico 1HZ75V se micro&lt;360 ou macro&lt;80 (**nao** usar hydrate como unico historico para treino meta — preferir Deriv / `--source auto`)
- MinIO: bucket `dl-models`; health live + `start_period`
- Loss-classifier: volume `loss-models/`; bootstrap opcional `python -m scripts.operations.train_loss_classifier`
- **Reset operacional:** `make docker-reset` — limpa pkls/TCN/volumes e gera seed `loss_bootstrap_synth` (`class_weight=balanced`, `min_child_samples=15`). Seed devolve **p_loss real** via `predict_proba` com `veto_ready=true` se `n_train>=ready_n`; sem `COLD_START` / sem `p_loss=0.50` neutro. Saida bootstrap exige buffer ≥**8** (`LOSS_BOOTSTRAP_EXIT_N`) com ≥1 WIN e ≥1 LOSS; primeiro fit promove so com `n>=LOSS_READY_N` e rejeita colapso (`collapsed_reject`, flag `collapsed` no `/predict`). LEARN loga `retrain_skipped_reason`. Floor FLIP **0.90**. `make docker-rebuild` so recarrega os containers meta/loss (preserva TCN e `meta_lgbm.pkl`).

## Relacao com o motor

Com `infra.enabled: true`, startup valida Redis/Timescale/MinIO (fail-fast). Mensagem operacional: `make docker-up-core|docker-up`. Detalhe de software: [`arquitetura.md`](arquitetura.md). Skill: `aether-infra-stack`.
