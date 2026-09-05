# DevOps & CloudOps Sênior — Aether Quantum Engine

SSOT técnico da stack híbrida local: motor no **host** (WSL/Conda, Python 3.13); sidecars em Docker profiles `core,ml`. Ops do dia a dia: [`infra-docker.md`](infra-docker.md). Arquitetura de camadas: [`engineering-architecture-senior.md`](engineering-architecture-senior.md). Runtime CPython: [`engineering-python-313-runtime.md`](engineering-python-313-runtime.md).

## 1. Docker e Compose

### Multi-stage (meta / loss)

- Builder: `python:3.13-slim` + toolchain mínima → virtualenv em `/opt/venv`.
- Runtime: copia só venv + app; `libgomp1` + `tini`; `USER aether` (UID 10001); limpa `apt` lists.
- Meta operacional de tamanho enxuto; LightGBM/OpenMP pode impedir &lt;200 MB estrito — não é gate hard de CI.
- **CMD exec-form** (PID 1 via `tini`): sinais SIGTERM/SIGINT chegam ao Uvicorn.
- Processo escuta `0.0.0.0`; publicação no host é `127.0.0.1:8005/8006`.

### Recursos e isolamento

- Compose **sem Swarm**: usar `mem_limit` / `cpus` (efetivos). Não depender de `deploy.resources`.
- `mem_swappiness: 0` nos sidecars (Redis, Timescale, MinIO, meta, loss) para preferir contenção de heap antes de swap WSL.
- Exit **137** → checar OOM (`dmesg`) e limites cgroup.
- `security_opt: [no-new-privileges:true]`; meta/loss `read_only` + `tmpfs /tmp`.
- Threading ML no container (obrigatório):

```text
OMP_NUM_THREADS=2
OPENBLAS_NUM_THREADS=2
MKL_NUM_THREADS=2
VECLIB_MAXIMUM_THREADS=2
NUMEXPR_NUM_THREADS=2
```

### Rede e health

- Binds: `127.0.0.1:6379|5432|9000|9001|8005|8006`.
- `depends_on` + `condition: service_healthy` (ex.: `minio-init` → MinIO; ML → MinIO).
- Make `docker-wait-healthy` permanece rede de segurança.
- Host: `infra/docker/host-prereq.sh` aplica `vm.overcommit_memory=1` (fork/COW Redis).

## 2. MinIO (SNSD)

- Single-node single-drive local (portas 9000/9001).
- Bucket SSOT: **`dl-models`** (`config/settings.json`).
- Bootstrap: serviço `minio-init` (`minio/mc`) cria bucket e ILM ~**7 dias** no prefixo `optuna/` (checkpoints intermediários).
- Consumidor valida integridade (ETag/MD5) antes de carregar artefato em memória.
- Segredos só via env (`.env`); least privilege nas keys de app quando aplicável.

## 3. Redis 7.4.2 Alpine

| Knob | Valor |
|------|--------|
| AOF | `appendonly yes` + `appendfsync everysec` |
| RDB | `save ""` (desligado) |
| `maxmemory` | `200mb` (headroom sob `mem_limit: 256m`) |
| `maxmemory-policy` | **`noeviction`** (falha explícita; sem drop silencioso da fila) |
| `io-threads` | `2` + `io-threads-do-reads yes` |

Settlement SSOT: ZSET **`settlement:queue:priority`**. **Proibido** migrar para Redis Streams/listas sem mandato explícito + migração testada.

Diagnóstico: `redis-cli info memory`, `slowlog get 10`.

## 4. TimescaleDB 2.28.2 (PG 16)

| Item | SSOT |
|------|------|
| Chunk | **1 day** (`ticks`, `ohlc_bars`) |
| Tuning (container 1g) | `shared_buffers=256MB`, `work_mem=16MB`, `checkpoint_completion_target=0.95` |
| Compressão | `segmentby=symbol`, `orderby=time DESC`, policy **7 days** |
| Retenção ticks | **30 days** |
| CRAG M5 | view `candle_m5` (analytics/ingest) — **não** substitui candles live Deriv |

`synchronous_commit=off` **não** é default; só trade-off experimental documentado para ticks, nunca para estado de settlement.

Diagnóstico idle-in-txn:

```sql
SELECT pid, now() - state_change AS duration, query
FROM pg_stat_activity
WHERE state = 'idle in transaction'
  AND (now() - state_change) > interval '5 seconds';
```

## 5. Matriz de triagem rápida

| Sintoma | Checar |
|---------|--------|
| Latência ticks | Redis slowlog; locks PG; MTU WSL |
| Contêiner morto | exit 137 / OOM; chunk Timescale vs RAM |
| CPU 100% sidecar | `OMP_NUM_THREADS`; loop Uvicorn; GIL/inferência |
| SETTLE estagnado | ZSET `settlement:queue:priority`; skill settlement |

## Anti-padrões

- Expor portas em `0.0.0.0` no host
- CMD shell-form engolindo SIGTERM
- `allkeys-lru` / eviction cega no Redis de settlement
- Trocar ZSET por Streams “por performance”
- Rewire do orquestrador para ler só CRAG (Deriv permanece SSOT de candle live)
- Dependência de `deploy.resources` fora de Swarm

## Paths

- `infra/docker/docker-compose.yml`
- `infra/docker/redis.conf`
- `infra/docker/minio-init.sh`
- `infra/docker/003_init-timescale.sql` / `004_timescale-lifecycle.sql` / `005_timescale_crags.sql`
- `infra/docker/meta-classifier/Dockerfile` / `loss-classifier/Dockerfile`
- Skills: `aether-devops-cloudops`, `aether-infra-stack`, `aether-redis-hiredis`, `aether-asyncpg-timescale`
- Rule: `aether-infra.mdc`
