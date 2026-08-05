# Documentação

| Documento | Conteúdo |
|-----------|----------|
| [../AGENTS.md](../AGENTS.md) | Entrada para agentes Cursor/LLM |
| [agent-coverage.md](agent-coverage.md) | Matriz 100%: doc + rule + skill por superficie |
| [arquitetura.md](arquitetura.md) | Arquitetura runtime: DL 34D, meta 43D, direção modular, quality gates, Soft Recovery, settlement |
| [structure.md](structure.md) | Layout do repositório e inventário de módulos Python em `app/src/` (**246**) |
| [medallion.md](medallion.md) | Metodologia: TCN × meta Z-Score, price zone, Kelly + Soft Recovery, SIDE_EQ, starvation |
| [sample-size-lln.md](sample-size-lln.md) | Lei dos Grandes Numeros: sample_size_policy, cold-start e anti vies dos pequenos numeros |
| [llm-trading-doctrine.md](llm-trading-doctrine.md) | Doutrina LLM/Cursor: 9 livros mapeados a gates, risco e anti-padroes de engenharia |
| [binary-senior-playbook.md](binary-senior-playbook.md) | Playbook trader senior: CALL/PUT/SKIP, catalogo gate_reason, knobs 30s (micro 60s) |
| [engineering-standards.md](engineering-standards.md) | QA: pre-commit, cobertura 100%, 300 linhas, commitlint, contribuicao |
| [engineering-orchestrator.md](engineering-orchestrator.md) | Ciclo do orquestrador, signature, locks, pos-settlement |
| [engineering-deep-learning.md](engineering-deep-learning.md) | DL 34D, labels, treino/run, meta offline, Triton |
| [engineering-settlement.md](engineering-settlement.md) | Fila Redis, tolerancia, profit_table, orphans |
| [engineering-settings-ssot.md](engineering-settings-ssot.md) | Mapa de `settings.json` e regra de knobs novos |
| [engineering-observability.md](engineering-observability.md) | Logger, dedupe, tags de log do ciclo |
| [engineering-logging-inventory.md](engineering-logging-inventory.md) | Mapa de fontes de log (runtime/scripts/infra) |
| [infra-docker.md](infra-docker.md) | Stack Docker hibrida: profiles `core/gpu/cpu/ml`, binds localhost, hydrate 120/600, smoke |
| [deriv-api.md](deriv-api.md) | Referência Deriv + integração PAT/OTP (retries Cloudflare/5xx) |
| [deriv-api-aether.md](deriv-api-aether.md) | Guia rápido Deriv para agentes (mapeamento Aether híbrido OTP/REST) |
| [deriv-indices-algorithm.md](deriv-indices-algorithm.md) | Algoritmo CSPRNG dos índices Drift |
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
| Domain | Risco Kelly + Soft Recovery (`soft_recovery_policy`), `RiskPolicy`, modelos, math, `side_equilibrium` |
| Infrastructure | Deriv (REST/WS com retry), Redis, Triton, MinIO, Timescale |
| Presentation | Logger terminal |

Regra: **domain** não importa application nem infrastructure. **Application** orquestra domain + adapters; implementações concretas vêm de `infra_factory`.

## Snapshot operacional (`config/settings.json`)

| Item | Valor |
|------|-------|
| Universo | `R_10` (âncora `R_10`) |
| DL | TCN, lookback **720**, micro **60 s**, macro **300 s**, `FEATURE_DIM=34`, label `ma_trend`, tensor `[1, 720, 34]` |
| Meta | LightGBM HTTP `:8005`, `META_FEATURE_DIM=43` (micro **60 s**); **opcional** para execução |
| Relógio | Micro **60 s** + macro **300 s** (1:5); contrato **30 s** (híbrido); assinatura legado `m5b:…;m5:…;m15:…` |
| Ciclo / assinatura | `cycle_interval_seconds` / `signature_boundary_seconds` = **60 s** |
| Execução | **Mandatória** (`mandatory_trade_each_cycle: true`); `force_trade_every_cycle: false`; `price_zone` alinha BUY→CALL / SELL→PUT |
| Fail-closed | Triton e meta **opcionais** nos settings atuais (`infra.triton.enabled/require_for_execution: false`; `require_meta_for_execution: false`) |
| Calibração | `neutral_half_width: 0.0` (zona neutra **off**); thresholds CALL/PUT **0.51/0.49**; override TCN macro se raw &gt;0.65 ou &lt;0.35 |
| Direção | Resolver modular (`checks` → `persistence` → `meta_edge` → `finalize`); SIDE_EQ antecipado; toxic escape **mantém** edge positivo |
| Persistence | Threshold **2** losses: tenta **flip** para o oposto (`side_eq_toxic_escape`); se o oposto também estiver saturado → skip |
| Quality gate | Pisos regulares de margem/edge/ADX **0.0** (esteira contínua); starvation a partir de **6** skips; edge decay a partir de **8** (`edge_decay_floor` → 0.0) |
| Recovery relax | `recovery_relax.edge_floor: -0.55` com `linear≥2` e pending |
| Discordance | `discordance_veto_enabled: false` (módulo `execution_direction_discordance` disponível) |
| Risco | Kelly EXPLORE (`fraction=0.08`, teto 3,5%) + Soft Recovery RECOVER (`max_safe_stake_pct=0.035`); SIDE_EQ LLN; stop win 2,60% (≥$100) / $10 (&lt;$100) |
| Settlement | Tolerância **90 s**, reconciliação passiva; pós-EXEC_EMPTY alinha fronteira (cap `exec_empty_retry_seconds`) |
| Watchdog | Stale tick **25 s** |
| Histórico treino | **23328** barras (@ granularidade de treino; micro 60 s no SSOT atual) |
| QA | Pre-commit: lint + testes **100%** cobertura (**305** `test_*.py`) + security; ≤300 linhas/arquivo |
