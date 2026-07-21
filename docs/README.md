# Documentação

| Documento | Conteúdo |
|-----------|----------|
| [arquitetura.md](arquitetura.md) | Arquitetura completa: pipeline runtime, DL 34D, meta 43D, quality gates (soft + HARD microestrutura), risco, settlement, config |
| [structure.md](structure.md) | Layout do repositório e inventário de módulos Python em `app/src/` (~226) |
| [medallion.md](medallion.md) | Metodologia quantitativa: scoring TCN × meta Z-Score, zonas compra/venda, sizing Kelly + Soft Recovery, side equilibrium (LLN), D-SQUEEZE |
| [infra-docker.md](infra-docker.md) | Stack Docker híbrida: profiles `core/gpu/ml`, hydrate/smoke, Triton GPU, meta-classifier |
| [deriv-api.md](deriv-api.md) | Referência Deriv + integração PAT/OTP |
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
| Application | Orquestração, DL, direção, quality gates, meta |
| Domain | Risco Kelly + Soft Recovery (`soft_recovery_policy`), `RiskPolicy`, modelos, math, `side_equilibrium` |
| Infrastructure | Deriv, Redis, Triton, MinIO, Timescale |
| Presentation | Logger terminal |

Regra: **domain** não importa application nem infrastructure. **Application** orquestra domain + adapters; implementações concretas vêm de `infra_factory`.

## Snapshot operacional (settings atuais)

| Item | Valor |
|------|-------|
| Universo | `R_10` (âncora `R_10`) |
| DL | TCN, lookback **72**, macro **600 s**, `FEATURE_DIM=34`, label `spot_forward`, tensor `[1, 72, 34]` |
| Meta | LightGBM HTTP `:8005`, `META_FEATURE_DIM=43` (micro **120 s**) |
| Relógio | Micro **120 s** + macro **600 s** (proporção **1:5**); assinatura legado `m5b:…;m5:…;m15:…` (nomes legados para 120/600) |
| Ciclo / assinatura | `cycle_interval_seconds` / `signature_boundary_seconds` = **120 s** |
| Execução | Seletiva (`price_zone`), contrato RISE_FALL **120 s** |
| Fail-closed | Triton obrigatório; meta **opcional** (`require_meta_for_execution: false`) |
| Calibração | `neutral_half_width: 0.04` (banda `[0.46, 0.54]`); thresholds **0.54/0.46**; override TCN macro se raw &gt;0.65 ou &lt;0.35 |
| Persistence / meta veto | Persistence **skip** (sem flip); `meta_veto_mode` none/soft/hard; `require_meta_for_execution: false` |
| Indicator gating | enabled; `adx_min` 0.20; `vol_ratio_min` 0.65; `veto_on_noise` false |
| Quality gate | Dual soft + HARD microestrutura + margem (`min_adx_threshold` 0.20; `min_direction_margin` 0.03; `mandatory_min_trade_score` 0.52; `min_validation_accuracy_gate` 0.63) |
| BB squeeze adaptativo | **desabilitado** |
| Loss protection | `min_direction_margin` 0.03; caps edge/Z 999 |
| Risco | Kelly EXPLORE (`fraction=0.08`, teto 3,5%) + Soft Recovery RECOVER (`max_safe_stake_pct=0.035`); side_equilibrium LLN; stop win 2,60% (≥$100) / $10 (&lt;$100) |
| Settlement | Tolerância **90 s**, reconciliação passiva |
| Watchdog | Stale tick **25 s** |
| Triton | `infer_timeout_seconds: 0.50`, `require_for_execution: true` |
| Histórico treino | **23328** barras (~162 dias @ 600 s) |
| Starvation | Decaimento a partir de **6** quality skips |
| QA | Pre-commit: lint + testes 100% cobertura (**287** `test_*.py`) + security |
