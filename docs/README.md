# Documentação

| Documento | Conteúdo |
|-----------|----------|
| [../AGENTS.md](../AGENTS.md) | Entrada para agentes Cursor/LLM |
| [../prompt-model.md](../prompt-model.md) | Contrato reutilizavel: DDD/hexagonal/TDD/DX para scaffold de novos repos |
| [agent-coverage.md](agent-coverage.md) | Matriz 100%: doc + rule + skill por superficie |
| [arquitetura.md](arquitetura.md) | Arquitetura runtime: DL 34D, meta 43D, fusao EV, signal_skip 1.1, Soft Recovery, settlement |
| [structure.md](structure.md) | Layout do repositório e inventário de módulos Python em `app/src/` (**246**) |
| [medallion.md](medallion.md) | Metodologia: TCN × meta Z-Score, price zone, Kelly + Soft Recovery, SIDE_EQ, starvation |
| [sample-size-lln.md](sample-size-lln.md) | Lei dos Grandes Numeros: sample_size_policy, cold-start e anti vies dos pequenos numeros |
| [llm-trading-doctrine.md](llm-trading-doctrine.md) | Doutrina LLM/Cursor: 9 livros mapeados a gates, risco e anti-padroes de engenharia |
| [binary-senior-playbook.md](binary-senior-playbook.md) | Playbook trader senior: CALL/PUT/SKIP, catalogo gate_reason, knobs M1 (micro 60s) |
| [engineering-standards.md](engineering-standards.md) | QA: pre-commit, cobertura 100%, 300 linhas, commitlint, contribuicao |
| [engineering-python-deps.md](engineering-python-deps.md) | Pins pip: anti-redundancia, Polars-only (pandas proibido), ABI numpy/torch |
| [engineering-repo-hygiene.md](engineering-repo-hygiene.md) | Higienizacao: ondas seguras, morto comprovado, never-delete |
| [engineering-surface-sync.md](engineering-surface-sync.md) | Fechamento: sync docs/rules/skills + pre-commit + anti-sujeira |
| [engineering-orchestrator.md](engineering-orchestrator.md) | Ciclo do orquestrador, signature, locks, pos-settlement |
| [engineering-deep-learning.md](engineering-deep-learning.md) | DL 34D, labels, treino/run, meta offline, inferência local |
| [engineering-settlement.md](engineering-settlement.md) | Fila Redis, tolerancia, profit_table, orphans |
| [engineering-settings-ssot.md](engineering-settings-ssot.md) | Mapa de `settings.json` e regra de knobs novos |
| [engineering-observability.md](engineering-observability.md) | Logger, dedupe, tags de log do ciclo |
| [engineering-logging-inventory.md](engineering-logging-inventory.md) | Mapa de fontes de log (runtime/scripts/infra) |
| [infra-docker.md](infra-docker.md) | Stack Docker hibrida: profiles `core/ml`, binds localhost, hydrate, smoke |
| [deriv-api.md](deriv-api.md) | Referência Deriv + integração PAT/OTP (retries Cloudflare/5xx) |
| [deriv-api-aether.md](deriv-api-aether.md) | Guia rápido Deriv para agentes (mapeamento Aether híbrido OTP/REST) |
| [deriv-indices-algorithm.md](deriv-indices-algorithm.md) | Volatility 75 (1s) Index `1HZ75V` (M15 / D1) |
| [CHANGELOG.md](CHANGELOG.md) | Histórico de versões |

Ponto de entrada do projeto: [README.md](../README.md). Agentes: [AGENTS.md](../AGENTS.md).

## Camadas DDD (resumo)

```
presentation  →  application  →  domain
                    ↓
              infrastructure (adapters)
```

| Camada | Papel |
|--------|-------|
| Application | Orquestração, DL, direção modular, quality gates, meta |
| Domain | Risco Kelly Single-Strike 1% + Soft Recovery (`soft_recovery_policy`), `RiskPolicy`, modelos, math, `side_equilibrium` |
| Infrastructure | Deriv (REST/WS com retry), Redis, MinIO, Timescale |
| Presentation | Logger terminal |

Regra: **domain** não importa application nem infrastructure. **Application** orquestra domain + adapters; implementações concretas vêm de `infra_factory`.

## Snapshot operacional (`config/settings.json`)

| Item | Valor |
|------|-------|
| Universo | `1HZ75V` (âncora `1HZ75V`) |
| DL | TCN, lookback **20**, micro **900 s** (M15), macro **86400 s** (D1), `FEATURE_DIM=34`, label `supertrend_atr`, tensor `[1, 20, 34]` |
| Meta | LightGBM HTTP `:8005`, `META_FEATURE_DIM=43` (micro **900 s**); **opcional** para execução |
| Relógio | Micro/MINI **900 s** (M15) + macro **86400 s** (D1); contrato ops **15 m (M15)**; label TCN **N=1** vela M15; ratio **1:96**; ciclo **900 s** |
| Ciclo / assinatura | `cycle_interval_seconds` / `signature_boundary_seconds` = **900 s** (sync fecho M15); `exec_empty_retry` **900 s** |
| Execução | `mandatory_trade_each_cycle: false`; `force_trade_every_cycle: false`; `invert_exec_side: false`; fusao EV + anti-loss microestrutura M15 + signal_skip 1.1 |
| Fail-closed | Meta **opcional** nos settings atuais (`require_meta_for_execution: false`); TCN eager/CUDA local |
| Calibração | Thresholds CALL/PUT **0.46/0.34**; override TCN macro se raw &gt;0.82 ou &lt;0.18 |
| Direção | Resolver modular com anti-loss microestrutura M15 (EMA slope 9/21, RSI momentum) |
| Risco | Kelly Single-Strike 1% (`fraction=0.08`, alvo 1% da banca em payout 0.85 em 1 tacada M15); Soft Recovery RECOVER |
| Settlement | Tolerância **600 s**, reconciliação passiva; pós-EXEC_EMPTY alinha fronteira |
| Watchdog | Stale tick **300 s** |
| Histórico treino | **100** barras diárias D1 |
| QA | Pre-commit: lint + testes **100%** cobertura + security; ≤300 linhas/arquivo |
