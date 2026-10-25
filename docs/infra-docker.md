# Infraestrutura Docker

Stack local **hibrida**: motor no host (Conda/WSL, Python 3.13, CUDA local) com inferencia TCN no processo do motor; persistencia, indicador e observabilidade em containers. Doutrina CloudOps: [`engineering-devops-cloudops-senior.md`](engineering-devops-cloudops-senior.md). Arquitetura: [`engineering-architecture-senior.md`](engineering-architecture-senior.md). SSOT operacional deste doc; atalho em [`infra/docker/README.md`](../infra/docker/README.md).

## Servicos

| Servico | Porta (localhost) | Profile | Limite tipico | Uso |
|---------|-------------------|---------|---------------|-----|
| Redis (`aether-core-cache-redis`) | `127.0.0.1:6379` | `core` | 256m | Estado, risco, `settlement:queue:priority` (AOF everysec, `maxmemory`/`noeviction`) |
| TimescaleDB (`aether-core-data-timescaledb`) | `127.0.0.1:5432` | `core` | 1g | Ticks + OHLC macro **86400 s** (D1) / micro **300 s** (M5); chunk 1d; CRAG `candle_m5` para analytics |
| MinIO (`aether-core-storage-minio`) | `127.0.0.1:9000` / `9001` | `core` | 512m | Checkpoints / TorchScript; imagem local `aether-core-storage-minio`; bucket `dl-models` |
| Indicador (`aether-market-signal-indicator`) | `127.0.0.1:9102` | `indicator` | 512m | Subsistema isolado de analise M5/M15/H1 e confluencia institucional |
| Prometheus (`aether-observe-metrics-prometheus`) | `127.0.0.1:9090` | `telemetry` | 256m | Coleta de metricas do host e dos containers |
| Grafana (`aether-observe-dashboards-grafana`) | `127.0.0.1:3000` | `telemetry` | 256m | Dashboards operacionais Aether e Sinteticos |

Hardening: `restart: unless-stopped`, log rotate 10m×3, binds em **127.0.0.1**, `mem_swappiness: 0`, `no-new-privileges`, `depends_on: service_healthy` onde aplicavel.

Logs de um servico: `make docker-logs DOCKER_SERVICE=<alias>`. Aliases Make → compose:

| Alias | Compose service |
|-------|-----------------|
| `redis`, `aether-core-cache-redis` | `aether-core-cache-redis` |
| `ts`, `timescale`, `timescaledb`, `aether-core-data-timescaledb` | `aether-core-data-timescaledb` |
| `minio`, `aether-core-storage-minio` | `aether-core-storage-minio` |
| `indicator`, `aether-market-signal-indicator` | `aether-market-signal-indicator` |
| `prom`, `prometheus`, `aether-observe-metrics-prometheus` | `aether-observe-metrics-prometheus` |
| `grafana`, `aether-observe-dashboards-grafana` | `aether-observe-dashboards-grafana` |

## Profiles e Make

| Target | Profiles | Quando usar |
|--------|----------|-------------|
| `make docker-up` | `core,telemetry,indicator` (padrao) | Stack unica: Redis/TS/MinIO + indicador + Prometheus + Grafana |
| `make docker-down` | — | Para os containers preservando volumes |
| `make docker-rebuild` | `indicator` | Reconstroi a imagem do indicador |
| `make docker-reset` | — | Sanitiza run e volumes, recriando a stack |

Pipeline `docker-up`: `host-prereq` → compose up → wait healthy → timescale-lifecycle → hydrate → smoke. Indicador e motor compartilham `aether-core-data-timescaledb:5432`, Prometheus e Grafana em `localhost:3000`. O indicador expoe metricas em `localhost:9102`; os dashboards de sinais ficam na pasta **Sinteticos**. `make app-train` garante a stack e diagnostica o setup institucional M15/M5.

## Variaveis (`.env`)

| Variavel | Padrao |
|----------|--------|
| `AETHER_DOCKER_HEALTH_TIMEOUT` | `300` |
| `DOCKER_PROFILES` / `COMPOSE_PROFILES` | `core,telemetry,indicator` |
| `AETHER_METRICS_HOST_IP` | Gateway Windows detectado no WSL por `make docker-up`; use `host-gateway` quando o motor rodar no WSL |

Settings app: `infra.redis.url`, `infra.timescale.dsn`, `infra.minio` — sempre **localhost** no hibrido.

## Redis / Timescale / MinIO

- Redis AOF `appendfsync everysec` (`config/redis.conf`); health com `start_period`
- Timescale: init `sql/003_*.sql` + lifecycle `sql/004_*.sql` (`ohlc_bars` compress `segmentby=symbol,granularity`, `orderby=time DESC, epoch DESC`); `sh/docker-hydrate.sh` apenas verifica a quantidade de OHLC alinhado e nunca fabrica velas. Para popular dados reais: `ensure_timescale.py` com a API Deriv.
- MinIO: bucket `dl-models`; health live + `start_period`
- `make docker-logs`: default servicos **running**; `DOCKER_LOGS_TAIL` default **200**
- Volume Timescale ja inicializado nao reaplica `002`/`004` no boot: `make docker-timescale-lifecycle` reaplica compress/CRAG; first-init limpo apos `docker-reset` / volume novo

## Relacao com o motor

Com `infra.enabled: true`, startup valida Redis/Timescale/MinIO (fail-fast). Mensagem operacional: `make docker-up`. Detalhe de software: [`arquitetura.md`](arquitetura.md). Skill: `aether-infra-stack`.
