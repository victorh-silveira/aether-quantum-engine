# Infraestrutura Docker

Stack local **hibrida**: motor no host (Conda/WSL, Python 3.13, CUDA local) com inferencia TCN **eager** no processo do motor; persistencia e sidecars ML em containers. Doutrina CloudOps: [`engineering-devops-cloudops-senior.md`](engineering-devops-cloudops-senior.md). Arquitetura: [`engineering-architecture-senior.md`](engineering-architecture-senior.md). SSOT operacional deste doc; atalho em [`infra/docker/README.md`](../infra/docker/README.md).

## Servicos

| Servico | Porta (localhost) | Profile | Limite tipico | Uso |
|---------|-------------------|---------|---------------|-----|
| Redis (`aether-core-cache-redis`) | `127.0.0.1:6379` | `core` | 256m | Estado, risco, `settlement:queue:priority` (AOF everysec, `maxmemory`/`noeviction`) |
| TimescaleDB (`aether-core-data-timescaledb`) | `127.0.0.1:5432` | `core` | 1g | Ticks + OHLC macro **86400 s** (D1) / micro **300 s** (M5); chunk 1d; CRAG `candle_m5` para analytics |
| MinIO (`aether-core-storage-minio`) | `127.0.0.1:9000` / `9001` | `core` | 512m | Checkpoints / TorchScript; imagem local `aether-core-storage-minio`; bucket `dl-models`; init + ILM `optuna/` ~7d |
| MinIO init (`aether-core-storage-minio-init`) | — | `core` | oneshot | Imagem `pgsty/mc`; cria bucket/ILM e **sai com Exit 0**; `Exited (0)` em `docker ps -a` e **sucesso**, nao falha. Meta/loss esperam `service_completed_successfully`. |
| Meta (`aether-ml-meta-classifier`) | `127.0.0.1:8005` | `ml` | 512m | LGBMRegressor **23D**; schema_hash; `/v2/predict_meta` aplica `label_scale` e clamp **[-1, +0.85]**; `/v1/learn` idempotente por `contract_id`; nao promove `meta_online_*` sobre `meta_lgbm.pkl` sem gate N>=32 e MAE; fit fora do loop |
| Loss (`aether-ml-loss-classifier`) | `127.0.0.1:8006` | `ml` | 512m | LGBMClassifier **24D** semantico + schema_hash; `/v1/learn` dedupe `contract_id`; pesos de recencia se N maduro; `degenerate` se colapso/ECE extremo (`veto_ready=false`, sem SKIP); T=2 young; saida do seed = `LOSS_BOOTSTRAP_EXIT_N` **4**; motor **FLIP** so apos auto_learn; young pe>=0.55 / mature pe>=0.58 |

Hardening: `restart: unless-stopped` (servicos longos), `aether-core-storage-minio-init` com `restart: "no"` (oneshot), log rotate 10m×3, binds em **127.0.0.1**, `mem_swappiness: 0`, `no-new-privileges`, `depends_on: service_healthy` / `service_completed_successfully` onde aplicavel, OMP*=2 nos ML.

Logs de um servico: `make docker-logs DOCKER_SERVICE=<alias>`. Aliases Make → compose:

| Alias | Compose service |
|-------|-----------------|
| `redis`, `aether-core-cache-redis` | `aether-core-cache-redis` |
| `ts`, `timescale`, `timescaledb`, `aether-core-data-timescaledb` | `aether-core-data-timescaledb` |
| `minio`, `aether-core-storage-minio` | `aether-core-storage-minio` |
| `meta`, `meta-classifier`, `aether-ml-meta-classifier` | `aether-ml-meta-classifier` |
| `loss`, `loss-classifier`, `aether-ml-loss-classifier` | `aether-ml-loss-classifier` |

## Profiles e Make

| Target | Profiles | Quando usar |
|--------|----------|-------------|
| `make docker-up` | `core,ml,telemetry,indicator` (padrao) | Stack unica: Redis/TS/MinIO + meta + loss + indicador + Prometheus + Grafana |
| `make docker-up-core` | `core` | So Redis/TS/MinIO |

Pipeline `docker-up`: `host-prereq` → compose up → wait healthy → timescale-lifecycle → hydrate (âncora de `config/settings.json`, atualmente R_25, micro/macro) → smoke. Indicador e motor compartilham `aether-core-data-timescaledb:5432`, Prometheus e Grafana em `localhost:3000`. O indicador expõe métricas em `localhost:9102`; os dashboards de sinais ficam na pasta **Sinteticos**. `make app-train` garante a stack e treina indicador M5/M15/H1 antes de TCN/meta/loss; no treino TCN, sincroniza histórico de todos os ativos em `deep_learning.train_symbols`.

Fluxo diario: `make app-train` garante a stack, treina indicador M5/M15/H1, bootstrap loss, TCN e meta dos indices M5 aprovados, valida checkpoints e recarrega os sidecars. `make docker-up` sobe a stack sem treinar. `make docker-rebuild` reconstroi imagens meta/loss/indicador e recarrega os arquivos locais **sem** apagar `data/dl` nem `indicator/data/indicator-models`. Rebuild **nao** chama `sanitize_fresh_run`. Reset destrutivo: `make docker-reset` (sanitiza TCN/loss/estado, mantem `meta_lgbm.pkl`, `down --volumes`). Sanitizacao total (inclui `meta_lgbm.pkl` e `data/dl/*.pth`): `make sanitize-run` ou etapa 0 de `launch-train.bat`. Smoke: processo meta pode subir sem `.pkl` (aviso); modelo so apos treino meta concluido.

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
| `DOCKER_PROFILES` / `COMPOSE_PROFILES` | `core,ml,telemetry` |
| `AETHER_METRICS_HOST_IP` | Gateway Windows detectado no WSL por `make docker-up`; use `host-gateway` quando o motor rodar no WSL |

Settings app: `infra.redis.url`, `infra.timescale.dsn`, `infra.minio`, `infra.meta_classifier`, `infra.loss_classifier` — sempre **localhost** no hibrido.

## Redis / Timescale / MinIO

- Redis AOF `appendfsync everysec` (`config/redis.conf`); health com `start_period`
- Timescale: init `sql/003_*.sql` + lifecycle `sql/004_*.sql` (`ohlc_bars` compress `segmentby=symbol,granularity`, `orderby=time DESC, epoch DESC`); `sh/docker-hydrate.sh` apenas verifica a quantidade de OHLC alinhado e nunca fabrica velas. Para popular dados reais: `launch-train` → `ensure_timescale.py` → seed Deriv **M5×5000 + D1×365** (timeout **900s**). A migração `009_contract_label_audit_m5.sql` atualiza a auditoria de contratos em volumes existentes.
- O stream grava cada M5 micro fechada em `ohlc_bars` sem bloquear a decisao por flush; gravacoes de ticks e contratos continuam em lote. Conferir `max(epoch)` para `granularity=300` antes de interpretar `contract_label_audit`: barras antigas nao sao reconstruidas pelo writer live.
- `make docker-logs`: default servicos **running** (exclui `minio-init` oneshot); `DOCKER_SERVICE=minio-init` para oneshot; `DOCKER_LOGS_TAIL` default **200**
- Volume Timescale ja inicializado nao reaplica `002`/`004` no boot: `make docker-timescale-lifecycle` reaplica compress/CRAG; first-init limpo apos `docker-reset` / volume novo
- MinIO: bucket `dl-models`; health live + `start_period`
- Loss-classifier: volume `loss-models/`; bootstrap opcional `python -m scripts.operations.train_loss_classifier`
- Nucleo compartilhado: `infra/docker/ml_common/` (schema_hash, vetor finito, persistencia atomica, dedupe `contract_id`) copiado nos Dockerfiles via `additional_contexts` (`COPY --from=ml_common`; Hadolint ignora DL3022 — contexto BuildKit nomeado, nao stage `FROM`)
- **Reset operacional:** `make docker-reset` — limpa pkls/TCN/volumes e gera seed `loss_bootstrap_live64` (`class_weight=balanced`, features na **escala live 24D**). Se `learn_buffer.pkl` tiver N≥**12** e WIN+LOSS, `train_loss_classifier.py` grava `loss_seed_real{n}` (`auto_learn=True`). Seed legado `loss_bootstrap_synth` saturava `p_loss` no live; o container **reseeda** bootstrap no startup. Bootstrap/young aplicam temperatura **T=2**; maduro (`n_train>=32`) usa temperatura do buffer (NLL). `veto_ready` se `n_train>=LOSS_READY_N` (**32**, SSOT `ready_n` **32**) e nao `degenerate`. Saida do seed live: `LOSS_BOOTSTRAP_EXIT_N` **4** (WIN+LOSS); **nao** esperar `READY_N` **32** para `auto_learn`. Retrain: `LOSS_RETRAIN_MIN_N` **12**, `LOSS_MIN_WIN_FOR_LOSS_RETRAIN` **4**. Sidecar `LOSS_VETO_P_LOSS_FLOOR` **0.58** e telemetria (motor ignora para SKIP). Meta: `retrain_min_n` **32**, fit huber, `label_scale` no predict, preferir `meta_lgbm` sobre `meta_online_*` salvo gate N>=32 e MAE buffer ≤2× `val_mae`; clamp **sempre** **[-1, +0.85]**. Motor: fallback meta com `meta_applied=false` e edge omitido (nao `0.0` falso); **FLIP** so apos auto_learn; young pe>=**0.55** / mature pe>=**0.58** (shrink N young; tape telemetria; sidecar T=2 se `n_train < LOSS_YOUNG_TEMP_N` **32**; sem HARD SKIP/Soft). `make docker-rebuild` recarrega meta/loss e reconstroi o indicador (preserva TCN, `meta_lgbm.pkl` e modelos do indicador).

## Relacao com o motor

Com `infra.enabled: true`, startup valida Redis/Timescale/MinIO (fail-fast). Mensagem operacional: `make docker-up-core|docker-up`. Detalhe de software: [`arquitetura.md`](arquitetura.md). Skill: `aether-infra-stack`.
