# Documentação

| Documento | Conteúdo |
|-----------|----------|
| [../AGENTS.md](../AGENTS.md) | Entrada para agentes Cursor/LLM |
| [agent-coverage.md](agent-coverage.md) | Matriz 100%: doc + rule + skill por superficie |
| [arquitetura.md](arquitetura.md) | Arquitetura runtime: DL 34D, meta 43D, fusao EV, signal_skip 1.1, Soft Recovery, settlement |
| [structure.md](structure.md) | Layout do repositório e inventário de módulos Python em `app/src/` (**246**) |
| [medallion.md](medallion.md) | Metodologia: TCN × meta Z-Score, price zone, Kelly + Soft Recovery, SIDE_EQ, starvation |
| [sample-size-lln.md](sample-size-lln.md) | Lei dos Grandes Numeros: sample_size_policy, cold-start e anti vies dos pequenos numeros |
| [llm-trading-doctrine.md](llm-trading-doctrine.md) | Doutrina LLM/Cursor: 9 livros mapeados a gates, risco e anti-padroes de engenharia |
| [binary-senior-playbook.md](binary-senior-playbook.md) | Playbook trader senior: CALL/PUT/SKIP, catalogo gate_reason, knobs M1 (micro 60s) |
| [engineering-standards.md](engineering-standards.md) | QA: pre-commit, cobertura 100%, 300 linhas, commitlint, contribuicao |
| [engineering-python-deps.md](engineering-python-deps.md) | Pins pip: anti-redundancia, dual-stack pandas/polars, ABI numpy/torch |
| [engineering-orchestrator.md](engineering-orchestrator.md) | Ciclo do orquestrador, signature, locks, pos-settlement |
| [engineering-deep-learning.md](engineering-deep-learning.md) | DL 34D, labels, treino/run, meta offline, inferência local |
| [engineering-settlement.md](engineering-settlement.md) | Fila Redis, tolerancia, profit_table, orphans |
| [engineering-settings-ssot.md](engineering-settings-ssot.md) | Mapa de `settings.json` e regra de knobs novos |
| [engineering-observability.md](engineering-observability.md) | Logger, dedupe, tags de log do ciclo |
| [engineering-logging-inventory.md](engineering-logging-inventory.md) | Mapa de fontes de log (runtime/scripts/infra) |
| [infra-docker.md](infra-docker.md) | Stack Docker hibrida: profiles `core/ml`, binds localhost, hydrate, smoke |
| [deriv-api.md](deriv-api.md) | Referência Deriv + integração PAT/OTP (retries Cloudflare/5xx) |
| [deriv-api-aether.md](deriv-api-aether.md) | Guia rápido Deriv para agentes (mapeamento Aether híbrido OTP/REST) |
| [deriv-indices-algorithm.md](deriv-indices-algorithm.md) | Universo `R_10` (Volatility 10 M1) e migracao |
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
| Infrastructure | Deriv (REST/WS com retry), Redis, MinIO, Timescale |
| Presentation | Logger terminal |

Regra: **domain** não importa application nem infrastructure. **Application** orquestra domain + adapters; implementações concretas vêm de `infra_factory`.

## Snapshot operacional (`config/settings.json`)

| Item | Valor |
|------|-------|
| Universo | `R_10` (âncora `R_10`) |
| DL | TCN, lookback **480**, micro **60 s**, macro **7200 s**, `FEATURE_DIM=34`, label `ma_trend`, tensor `[1, 480, 34]` |
| Meta | LightGBM HTTP `:8005`, `META_FEATURE_DIM=43` (micro **60 s**); **opcional** para execução |
| Relógio | Micro/MINI **60 s** + macro **7200 s**; contrato ops **5 m (M5)**; label TCN **N** ∈ {15,20,…,60} (**SSOT atual N=55**); ratio **1:120**; assinatura legado `m5b:…;m5:…;m15:…` |
| Ciclo / assinatura | `cycle_interval_seconds` / `signature_boundary_seconds` = **60 s** (sync fecho M1); `exec_empty_retry` **60 s** |
| Execução | `mandatory_trade_each_cycle: false`; `force_trade_every_cycle: false`; `invert_exec_side: false`; fusao EV + signal_skip 1.1 (quality gate amplo **fora**) |
| Fail-closed | Meta **opcional** nos settings atuais (`require_meta_for_execution: false`); TCN eager/CUDA local |
| Calibração | `neutral_half_width: 0.0` (zona neutra **off**); thresholds CALL/PUT **0.62/0.38**; override TCN macro se raw &gt;0.65 ou &lt;0.35 |
| Direção | Resolver modular (`checks` → `persistence` → `meta_edge` → `finalize`); SIDE_EQ antecipado; toxic escape **mantém** edge positivo |
| Persistence | Threshold **2** losses: tenta **flip** para o oposto (`side_eq_toxic_escape`); se o oposto também estiver saturado → skip |
| Quality gate | Pisos regulares de margem/edge/ADX **0.0** (esteira contínua); starvation a partir de **6** skips; edge decay a partir de **8** (`edge_decay_floor` → 0.0) |
| Recovery relax | `recovery_relax.edge_floor: -0.55` com `linear≥2` e pending |
| Discordance | `discordance_veto_enabled: false` (módulo `execution_direction_discordance` disponível) |
| Risco | Kelly EXPLORE (`fraction=0.08`, piso **0.25%** / teto **5%**) + Soft Recovery RECOVER (`max_safe_stake_pct=0.05`, payout **0.72**); stop-win Kelly **4 ciclos/1h**; stop win 3% (≥$100) / $10 (&lt;$100) |
| Settlement | Tolerância **90 s**, reconciliação passiva; pós-EXEC_EMPTY alinha fronteira (cap `exec_empty_retry_seconds`) |
| Watchdog | Stale tick **300 s** |
| Histórico treino | **2000** barras micro M1; sync lean no treino (macro≤128, mini=0) |
| QA | Pre-commit: lint + testes **100%** cobertura (**305** `test_*.py`) + security; ≤300 linhas/arquivo |
