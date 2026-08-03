# Documentação

| Documento | Conteúdo |
|-----------|----------|
| [arquitetura.md](arquitetura.md) | Arquitetura runtime: DL 34D, meta 43D, direção modular, quality gates, Soft Recovery, settlement |
| [structure.md](structure.md) | Layout do repositório e inventário de módulos Python em `app/src/` (**246**) |
| [medallion.md](medallion.md) | Metodologia: TCN × meta Z-Score, price zone, Kelly + Soft Recovery, SIDE_EQ, starvation |
| [sample-size-lln.md](sample-size-lln.md) | Lei dos Grandes Numeros: sample_size_policy, cold-start e anti vies dos pequenos numeros |
| [infra-docker.md](infra-docker.md) | Stack Docker híbrida: profiles `core/gpu/ml`, hydrate/smoke, Triton `R_10`, meta-classifier |
| [deriv-api.md](deriv-api.md) | Referência Deriv + integração PAT/OTP (retries Cloudflare/5xx) |
| [deriv-api-aether.md](deriv-api-aether.md) | Guia rápido Deriv para agentes (mapeamento Aether híbrido OTP/REST) |
| [deriv-indices-algorithm.md](deriv-indices-algorithm.md) | Algoritmo CSPRNG dos índices Drift |
| [CHANGELOG.md](CHANGELOG.md) | Histórico de versões |

Ponto de entrada do projeto: [README.md](../README.md).

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
| DL | TCN, lookback **72**, macro **600 s**, `FEATURE_DIM=34`, label `spot_forward`, tensor `[1, 72, 34]` |
| Meta | LightGBM HTTP `:8005`, `META_FEATURE_DIM=43` (micro **120 s**); **opcional** para execução |
| Relógio | Micro **120 s** + macro **600 s** (1:5); assinatura legado `m5b:…;m5:…;m15:…` |
| Ciclo / assinatura | `cycle_interval_seconds` / `signature_boundary_seconds` = **120 s** |
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
| Histórico treino | **23328** barras (~162 dias @ 600 s) |
| QA | Pre-commit: lint + testes **100%** cobertura (**305** `test_*.py`) + security; ≤300 linhas/arquivo |
