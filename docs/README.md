# Documentação

| Documento | Conteúdo |
|-----------|----------|
| [arquitetura.md](arquitetura.md) | Arquitetura completa: pipeline runtime, DL 34D, meta 43D, quality gates, risco, settlement, config |
| [structure.md](structure.md) | Layout do repositório e inventário de módulos Python em `app/src/` |
| [medallion.md](medallion.md) | Metodologia quantitativa: scoring TCN × meta Z-Score, esteira contínua, soft recovery, Kelly, D-SQUEEZE |
| [infra-docker.md](infra-docker.md) | Stack Docker (Redis, Timescale, MinIO, Triton, meta-classifier) |
| [deriv-api.md](deriv-api.md) | Referência Deriv + integração PAT/OTP |
| [deriv-indices-algorithm.md](deriv-indices-algorithm.md) | Algoritmo CSPRNG dos índices Drift |
| [CHANGELOG.md](CHANGELOG.md) | Histórico de versões |

Ponto de entrada do projeto: [README.md](../README.md).

## Camadas DDD (resumo)

```
presentation  →  application  →  domain
                    ↓
              infrastructure (ports/adapters)
```

| Camada | Papel |
|--------|-------|
| Application | Orquestração, DL, direção, quality gates, meta (~140 módulos) |
| Domain | Risco Kelly/D'Alembert, `RiskPolicy`, modelos, math (~29 módulos) |
| Infrastructure | Deriv, Redis, Triton, MinIO, Timescale (~49 módulos) |
| Presentation | Logger terminal (1 módulo) |

Regra: **domain** não importa application nem infrastructure. **Application** orquestra domain + ports; implementações concretas vêm de `infra_factory`.

## Snapshot operacional (settings atuais)

| Item | Valor |
|------|-------|
| Universo | `RDBEAR`, `RDBULL` (âncora `RDBULL`) |
| DL | TCN, lookback 48, M15, `FEATURE_DIM=34`, label `ma_trend` |
| Meta | LightGBM HTTP `:8005`, `META_FEATURE_DIM=43` |
| Execução | Esteira mandatária, contrato RISE_FALL 60 s |
| Fail-closed | Meta + Triton obrigatórios para execução |
| Risco | Kelly `fraction=0.005`, teto 3,5%, soft recovery, stop win 2,60% |
| QA | Pre-commit: lint + testes 100% cobertura + security |
