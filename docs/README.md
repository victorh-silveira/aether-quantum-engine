# Documentação

| Documento | Conteúdo |
|-----------|----------|
| [../AGENTS.md](../AGENTS.md) | Entrada para agentes Cursor/LLM |
| [../prompt-model.md](../prompt-model.md) | Contrato reutilizavel: DDD/hexagonal/TDD/DX para scaffold de novos repos |
| [agent-coverage.md](agent-coverage.md) | Matriz 100%: doc + rule + skill por superficie |
| [arquitetura.md](arquitetura.md) | Arquitetura runtime: DL 14D, meta 23D, fusao EV, signal_skip 1.1, Soft Recovery, settlement |
| [engineering-architecture-senior.md](engineering-architecture-senior.md) | Doutrina sênior: host Python 3.13, DDD/hexagonal, asyncio/CUDA, Polars SSOT, sidecars ML, Docker core/ml, QA |
| [structure.md](structure.md) | Layout do repositório e inventário de módulos Python em `app/src/` (**246**) |
| [medallion.md](medallion.md) | Metodologia: TCN × meta Z-Score, price zone, Kelly + Soft Recovery, SIDE_EQ, starvation |
| [sample-size-lln.md](sample-size-lln.md) | Lei dos Grandes Numeros: sample_size_policy, cold-start e anti vies dos pequenos numeros |
| [llm-trading-doctrine.md](llm-trading-doctrine.md) | Doutrina LLM/Cursor: 9 livros mapeados a gates, risco e anti-padroes de engenharia |
| [binary-senior-playbook.md](binary-senior-playbook.md) | Playbook trader senior: CALL/PUT/SKIP, catalogo gate_reason, knobs M5 (micro 300s) |
| [engineering-standards.md](engineering-standards.md) | QA: pre-commit, cobertura 100%, 300 linhas, commitlint, contribuicao |
| [../.github/README.md](../.github/README.md) | CI Python/Docker/Shell/Workflows (steps `Area \| Stage`); JSON/YAML no job Python; release |
| [../linters/README.md](../linters/README.md) | Hooks locais: Ruff, pytest, Bandit, commitlint, semantic-release |
| [engineering-python-deps.md](engineering-python-deps.md) | SSOT deps sênior: pins, websockets/httpx, Polars, MinIO, ML, QA |
| [engineering-repo-hygiene.md](engineering-repo-hygiene.md) | Higienizacao: ondas seguras, morto comprovado, never-delete |
| [engineering-surface-sync.md](engineering-surface-sync.md) | Fechamento: sync docs/rules/skills + pre-commit + anti-sujeira |
| [engineering-orchestrator.md](engineering-orchestrator.md) | Ciclo do orquestrador, signature, locks, pos-settlement |
| [engineering-deep-learning.md](engineering-deep-learning.md) | DL 14D, labels, treino/run, meta offline, inferência local |
| [engineering-settlement.md](engineering-settlement.md) | Fila Redis, tolerancia, profit_table, orphans |
| [engineering-settings-ssot.md](engineering-settings-ssot.md) | Mapa de `settings.json` e regra de knobs novos |
| [engineering-observability.md](engineering-observability.md) | Logger, dedupe, tags de log do ciclo |
| [engineering-logging-inventory.md](engineering-logging-inventory.md) | Mapa de fontes de log (runtime/scripts/infra) |
| [engineering-python-313-runtime.md](engineering-python-313-runtime.md) | Runtime CPython 3.13: GIL, GC, asyncio, Polars, bordas Redis/Timescale/Torch |
| [engineering-devops-cloudops-senior.md](engineering-devops-cloudops-senior.md) | DevOps/CloudOps: Compose, Redis, Timescale, MinIO, multi-stage, triagem OOM/OMP |
| [infra-docker.md](infra-docker.md) | Stack Docker hibrida: profiles `core/ml`, binds localhost, hydrate, smoke |
| [deriv-api.md](deriv-api.md) | Referência Deriv + integração PAT/OTP (retries Cloudflare/5xx) |
| [deriv-api-aether.md](deriv-api-aether.md) | Guia rápido Deriv para agentes (mapeamento Aether híbrido OTP/REST) |
| [deriv-indices-algorithm.md](deriv-indices-algorithm.md) | Volatility 75 (1s) Index `1HZ75V` (M5 / D1) |
| [CHANGELOG.md](CHANGELOG.md) | Histórico de versões |

Ponto de entrada do projeto: [README.md](../README.md). Agentes: [AGENTS.md](../AGENTS.md).

## Camadas DDD (resumo)

Doutrina completa: [engineering-architecture-senior.md](engineering-architecture-senior.md).

```
presentation  →  application  →  domain
                    ↓
              ports → infrastructure (adapters)
```

Motor no **host** (Python 3.13 / asyncio / CUDA); Redis, Timescale, MinIO, meta e loss em Docker.

| Camada | Papel |
|--------|-------|
| Application | Orquestração, DL, direção modular, quality gates, meta via ports |
| Domain | Risco Kelly Single-Strike 4.31% + Soft Recovery (`soft_recovery_policy`), `RiskPolicy`, modelos, math, `side_equilibrium` (sem I/O) |
| Infrastructure | Deriv (REST/WS com retry), asyncpg, Redis, MinIO, sidecars HTTP |
| Presentation | Logger terminal (Rich); composition root |

Regra: **domain** não importa application nem infrastructure. **Application** orquestra domain + adapters; implementações concretas vêm de `infra_factory`. Event loop: offload de PyTorch/Polars pesado.

## Snapshot operacional (`config/settings.json`)

| Item | Valor |
|------|-------|
| Universo | `1HZ75V` (âncora `1HZ75V`) |
| DL | TCN, lookback **30**, micro **300 s** (500 velas M5), macro **86400 s** (365 velas D1), `FEATURE_DIM=14`, label `quantum_multi_barrier`, tensor `[1, 30, 14]` |
| Meta | LightGBM HTTP `:8005`, `META_FEATURE_DIM=23` (micro **300 s**); **opcional** para execução |
| Relógio | Micro/MINI **300 s** (M5) + macro **86400 s** (D1); contrato ops **5 m (M5)**; label TCN **N=1** vela M5; ratio **1:288**; ciclo **120 s** |
| Ciclo / assinatura | `cycle_interval_seconds` / `signature_boundary_seconds` = **300 s** (sync fecho M5); `exec_empty_retry` **120 s** |
| Execução | `mandatory_trade_each_cycle: false`; `force_trade_every_cycle: false`; `invert_exec_side: false`; fusao EV + anti-loss microestrutura M5 + signal_skip 1.1 |
| Fail-closed | Meta **opcional** nos settings atuais (`require_meta_for_execution: false`); TCN eager/CUDA local |
| Calibração | Thresholds CALL/PUT **0.565/0.435**; clamp Cal em `[raw±0.08]` antes da zona neutra; modo `raw_extreme` |
| Direção | Resolver modular com anti-loss microestrutura M5 (ancora hibrida, EMA slope 9/21, RSI momentum) |
| Risco | Kelly Single-Strike 4.31% (`fraction=0.08`, alvo 4.31% da banca em payout 0.85 em 1 tacada M5); Soft Recovery RECOVER |
| Settlement | Tolerância **600 s**, reconciliação passiva; pós-EXEC_EMPTY alinha fronteira |
| Watchdog | Stale tick **300 s** |
| Histórico treino | **100** barras diárias D1 |
| QA | Pre-commit: lint + testes **100%** cobertura + security; ≤300 linhas/arquivo |
