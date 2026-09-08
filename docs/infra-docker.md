# Infraestrutura Docker

Stack local **hibrida**: motor no host (Conda/WSL, Python 3.13, CUDA local) com inferencia TCN **eager** no processo do motor; persistencia e sidecars ML em containers. Doutrina CloudOps: [`engineering-devops-cloudops-senior.md`](engineering-devops-cloudops-senior.md). Arquitetura: [`engineering-architecture-senior.md`](engineering-architecture-senior.md). SSOT operacional deste doc; atalho em [`infra/docker/README.md`](../infra/docker/README.md).

## Servicos

| Servico | Porta (localhost) | Profile | Limite tipico | Uso |
|---------|-------------------|---------|---------------|-----|
| Redis | `127.0.0.1:6379` | `core` | 256m | Estado, risco, `settlement:queue:priority` (AOF everysec, `maxmemory`/`noeviction`) |
| TimescaleDB | `127.0.0.1:5432` | `core` | 1g | Ticks + OHLC macro **86400 s** (D1) / micro **300 s** (M5); chunk 1d; CRAG `candle_m5` analytics |
| MinIO | `127.0.0.1:9000` / `9001` | `core` | 512m | Checkpoints / TorchScript; bucket `dl-models`; `minio-init` + ILM `optuna/` ~7d |
| MinIO init (`aether-minio-init`) | — | `core` | oneshot | Cria bucket/ILM e **sai com Exit 0**; `Exited (0)` em `docker ps -a` e **sucesso**, nao falha. Meta/loss esperam `service_completed_successfully`. |
| Meta (`aether-meta-classifier`) | `127.0.0.1:8005` | `ml` | 512m | LGBMRegressor **23D**; `/v2/predict_meta` + `/v1/learn` online (`META_RETRAIN_MIN_N` / `retrain_min_n` **32**); ignora `meta_online_*` tiny; buffer `meta_learn_buffer.pkl` |
| Loss (`aether-loss-classifier`) | `127.0.0.1:8006` | `ml` | 512m | LGBMClassifier **24D**; buffer `learn_buffer.pkl` no volume; `/learn` + retrain **a cada trade** (WIN+LOSS no buffer; `LOSS_RETRAIN_MIN_N` **1**); saida bootstrap `LOSS_BOOTSTRAP_EXIT_N` **8**; motor HARD SKIP se `p_loss >= 0.90` |

Hardening: `restart: unless-stopped` (servicos longos), `minio-init` com `restart: "no"` (oneshot), log rotate 10m×3, binds em **127.0.0.1**, `mem_swappiness: 0`, `no-new-privileges`, `depends_on: service_healthy` / `service_completed_successfully` onde aplicavel, OMP*=2 nos ML.

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
| `POST /v2/predict_meta` | Vetor **23D** → `predicted_payoff_edge` |

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
- Timescale: init `003_*.sql` + lifecycle `004_*.sql` (`ohlc_bars` compress `segmentby=symbol,granularity`, `orderby=time DESC, epoch DESC`); hydrate sintetico `1HZ75V` M5(300s)×500 e D1(86400s)×365 se micro&lt;400 ou macro&lt;200 (**smoke only** — treino meta: `launch-train` → `ensure_timescale` seed Deriv **M5×5000 + D1×365**, timeout **900s**, persist em lote)
- `make docker-logs`: default servicos **running** (exclui `minio-init` oneshot); `DOCKER_SERVICE=minio-init` para oneshot; `DOCKER_LOGS_TAIL` default **200**
- Volume Timescale ja inicializado nao reaplica `002`/`004` no boot: `make docker-timescale-lifecycle` reaplica compress/CRAG; first-init limpo apos `docker-reset` / volume novo
- MinIO: bucket `dl-models`; health live + `start_period`
- Loss-classifier: volume `loss-models/`; bootstrap opcional `python -m scripts.operations.train_loss_classifier`
- **Reset operacional:** `make docker-reset` — limpa pkls/TCN/volumes e gera seed `loss_bootstrap_live64` (`class_weight=balanced`, features na **escala live 24D**). Seed legado `loss_bootstrap_synth` (Gaussiana N(0,1) com `y=LOSS` se `f0+0.3*f7>0`) saturava `p_loss` ~0.85–0.97 no live (`direction_margin`∈[0,1] = OOD); o container **reseeda** esse bootstrap no startup. Bootstrap aplica temperatura **T=2** no `predict_proba`. `veto_ready` se `n_train>=LOSS_READY_N` (**8**, SSOT `ready_n` **8**). Meta: `retrain_min_n` **32**, fit huber, preferir `meta_lgbm` sobre `meta_online_*` com n baixo; clamp edge online frio em **[-1, +0.85]**. Motor: HARD SKIP por `P_LOSS` (`hard_p_loss_floor` **0.90**, sem FLIP/Soft). `make docker-rebuild` so recarrega meta/loss (preserva TCN e `meta_lgbm.pkl`).

## Relacao com o motor

Com `infra.enabled: true`, startup valida Redis/Timescale/MinIO (fail-fast). Mensagem operacional: `make docker-up-core|docker-up`. Detalhe de software: [`arquitetura.md`](arquitetura.md). Skill: `aether-infra-stack`.
