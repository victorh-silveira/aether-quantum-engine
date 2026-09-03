# Arquitetura sênior — host Python 3.13, DDD/hexagonal, ML e infra híbrida

Doutrina de engenharia do **Aether Quantum Engine**. Runtime operacional (símbolo, M5, Kelly, gates) permanece em [`llm-trading-doctrine.md`](llm-trading-doctrine.md) e `config/settings.json`. Visão de pipeline: [`arquitetura.md`](arquitetura.md).

## 1. Princípio de implantação

O motor (`app/run.py` / `train.py`) roda **direto no host** (WSL Linux / Conda `deriv-api`, Python **3.13**) para:

- evitar sobrecarga de virtualização de rede no hot path;
- acesso direto a drivers **NVIDIA/CUDA** no host;
- manter o event loop asyncio próximo do WebSocket Deriv.

Sidecars (Redis, TimescaleDB, MinIO, meta `:8005`, loss `:8006`) ficam em Docker profiles `core,ml` com binds `127.0.0.1`.

## 2. Camadas DDD / hexagonal

```
                    ┌─────────────────────────┐
                    │ Presentation / Inbound  │
                    │ (WS Deriv, Rich UI CLI) │
                    └────────────┬────────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │   Application Layer   │
                     │  (Trading/Orchestr.)  │
                     └─────┬───────────┬─────┘
                           │           │
         ┌─────────────────┘           └─────────────────┐
         ▼                                               ▼
┌──────────────────┐                           ┌───────────────────┐
│   Domain Layer   │                           │ Outbound Ports    │
│  (Pure Entities, │                           │ (Interfaces: DB,  │
│ Value Objects,   │                           │  ML Models, WS)   │
│ Invariants, Risk)│                           └─────────┬─────────┘
└──────────────────┘                                     │
                                                         ▼
                                               ┌───────────────────┐
                                               │   Infrastructure  │
                                               │ (asyncpg, Polars, │
                                               │ Redis, Sidecars)  │
                                               └───────────────────┘
```

| Camada | Pasta | Contrato |
|--------|-------|----------|
| Presentation / inbound | `presentation/`, bootstrap WS | Rich CLI, composition root, logs semânticos |
| Application | `application/services/` | Orquestra ciclo, consulta domínio, invoca modelos via ports |
| Domain | `domain/` | Entidades/VOs/invariantes/risco **sem I/O** |
| Outbound ports | Protocols na application | Contratos DB, state, ML, market |
| Infrastructure | `infrastructure/`, `infra/docker/` | Adapters: asyncpg, Redis, MinIO, httpx, websockets, sidecars |

**Domain puro:** não importa application nem infrastructure. Preferir dataclasses imutáveis / `__slots__` onde o hot path exigir. Kelly, Soft Recovery e invariantes matemáticas vivem no domínio.

**Application:** pipeline reativo — consome ticks/barras, consulta domínio, inferência, despacha ordens só via ports abstratas.

## 3. Asyncio e event loop (Python 3.13)

- Não bloquear o loop principal com inferência PyTorch/CUDA nem transformações pesadas NumPy/Polars: delegar a `asyncio.to_thread` / `ThreadPoolExecutor` / subprocesso dedicado.
- Hot path live: `predict_symbol_decision_async` chama `eager_local_predict` via `asyncio.to_thread`.
- Preferir `torch.inference_mode()` / `no_grad`, tensores pré-alocados; avaliar `torch.compile(mode="reduce-overhead")` e `pin_memory()` sem travar o runtime.
- Estruturar tarefas com `asyncio.TaskGroup`, tratamento estrito de `CancelledError` e shutdown gracioso de conexões WS/HTTP.
- Circuit-breakers de processo: latência/SLA ou taxa de perdas acima do limiar → modo defensivo (rejeitar novas posições; drenar existentes). Stop-win / `EXEC_PAUSE` e gates de execução são a válvula operacional SSOT.
- SSOT JSON: I/O em [`app/settings_io.py`](../app/settings_io.py); parsers fail-closed em `domain/config_knobs.py` (sem `path.open` no domínio).

## 4. Host / WSL / CUDA (DevOps)

- `.wslconfig`: memória estrita, swap adequado, processors fixados quando necessário.
- Afinidade de CPU (`taskset` / `numactl`) para isolar cores do motor asyncio.
- Alinhar driver NVIDIA do Windows host com libs CUDA do Conda/WSL (`libcuda.so`) sem conflito de bibliotecas dinâmicas.
- Segredos Deriv/MinIO só via env / secret store local — nunca em manifests Conda, commits ou docs.

## 5. Machine Learning

| Papel | Onde | Contrato |
|-------|------|----------|
| Latência crítica TCN/LSTM/GRU | Host PyTorch CUDA | Batch 1, eager/`to_thread`, checkpoint local `data/dl/` |
| Meta LightGBM 43D | Sidecar FastAPI `:8005` | HTTP opcional conforme settings |
| Loss-classifier | Sidecar FastAPI `:8006` | Soft/FLIP conforme doutrina de gates |
| Optuna / tuning | Offline / background | Não disputar VRAM com inferência live |

- Cliente `httpx.AsyncClient` com pool keep-alive, timeout agressivo e fallback no motor se sidecar atrasar.
- Preferir payloads compactos (arrays) quando o contrato HTTP permitir; não inflar JSON no hot path.
- Artefatos versionados no MinIO (`.pt`, dumps LightGBM/joblib, scalers). Validar integridade (hash) antes de carregar em memória quando o pipeline de hydrate/deploy o exigir.
- Sidecars: Dockerfiles enxutos; `OMP_NUM_THREADS` dimensionado; `cgroups` (CPU/mem) para não saturar o host do motor.

## 6. Dados: Polars SSOT + NumPy

- **Proibido** pandas / `to_pandas` / dual-stack.
- Preferir LazyFrame para features em janela; materializar só na borda tensorial.
- Conversão controlada Polars → NumPy → torch; buffers circulares para ticks (não recriar DataFrame a cada tick).
- Tipagem numérica enxuta (Float32 quando suficiente).
- Runtime: `POLARS_MAX_THREADS` para não monopolizar núcleos vs event loop.
- Observar RSS do processo em sessões longas (anti OOM).

## 7. Mercado: websockets + httpx

- Heartbeat WS com `ping_interval` / `ping_timeout`; reconexão com backoff + jitter e re-subscribe atômico.
- Backpressure: preferir estado mais recente (lossy em ticks intermediários) se a fila acumular.
- REST Deriv: retries idempotentes, rate-limit por headers; auth híbrida PAT+OTP.
- Kernel TCP keep-alive para detectar link silencioso antes do timeout de aplicação.

## 8. Infra Docker local

| Serviço | Uso sênior Aether |
|---------|-------------------|
| Redis 7.4 | Estado efêmero + fila de settlement SSOT **ZSET** `settlement:queue:priority` (idempotente); AOF `everysec` |
| TimescaleDB / PG 16 | Hypertables, compressão, retenção; inserts em lote via `asyncpg`; gravação desacoplada da decisão |
| MinIO | Checkpoints e artefatos ML |

Compose: healthchecks (`redis-cli ping`, `pg_isready`, liveness MinIO); boot com `depends_on` + `service_healthy`. Volumes em storage rápido do host. Tuning `max_connections` / `shared_buffers` / `work_mem` alinhado ao pool asyncpg.

## 9. QA, segurança e observabilidade

| Prática | Dev | DevOps / SRE |
|---------|-----|--------------|
| pre-commit | Ruff + limpeza; bloquear artefatos grandes (`.pt` / `.parquet`) | Hooks versionados no repo |
| Ruff | Substitui Black/Flake8/isort; regras estritas | Gate rápido antes dos testes |
| pytest | Cobertura **100%** linhas/branches em `app/src`; domain sem I/O | CI fail-closed; espelho DDD |
| Bandit / pip-audit | Evitar deserialização insegura de origens não confiáveis | Bloquear CVE HIGH/CRITICAL |
| Rich | UI de terminal; não bloquear o loop | Daemon/CI: sem ANSI; logs estruturados |

Detalhes de gates: [`engineering-standards.md`](engineering-standards.md). Logging: [`engineering-observability.md`](engineering-observability.md).

## 10. Nunca

- Colocar I/O ou imports de infra no `domain/`
- Bloquear o event loop com CUDA/Polars pesado no hot path
- Expor portas Docker fora de `127.0.0.1`
- Commitar tokens Deriv / credenciais MinIO
- Introduzir pandas ou segunda lib de DataFrame
- Afrouxar cobertura 100% ou limite de 300 linhas para “passar o hook”
- Substituir a fila ZSET de settlement sem mandato e migração testada

## Referências

- Skill: `aether-architecture-senior`
- Rule: `aether-architecture-senior.mdc`
- Contrato cross-repo: [`../prompt-model.md`](../prompt-model.md)
- Infra: [`infra-docker.md`](infra-docker.md) + skill `aether-infra-stack`
- DL: [`engineering-deep-learning.md`](engineering-deep-learning.md) + skill `aether-dl-train`
- Deriv: [`deriv-api-aether.md`](deriv-api-aether.md) + skill `aether-deriv-connect`
