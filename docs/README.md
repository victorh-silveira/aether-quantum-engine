# Documentação

| Documento | Conteúdo |
|-----------|----------|
| [structure.md](structure.md) | Layout do repositório, **inventário completo dos 209 módulos Python** em `app/src/`, scripts, testes, pipeline de execução |
| [arquitetura.md](arquitetura.md) | Fluxos técnicos ao vivo: esteira mandatária, barreira atômica, fila Redis de settlement, DL, Triton, meta-regressor 39D, AntiTrendLock |
| [medallion.md](medallion.md) | Filosofia quantitativa: scoring TCN × meta Z-Score, esteira contínua, recovery Martingale, Kelly, D-SQUEEZE, meta 2,60% por sessão |
| [deriv-indices-algorithm.md](deriv-indices-algorithm.md) | Algoritmo CSPRNG dos índices Deriv Drift e estratégia do motor |
| [infra-docker.md](infra-docker.md) | Stack Docker (5 containers), Triton, meta-regressor, Redis pipeline, StateManager |
| [deriv-api.md](deriv-api.md) | Referência Deriv + integração PAT/OTP, propostas atômicas por sub-lote |
| [CHANGELOG.md](CHANGELOG.md) | Histórico de versões |

Ponto de entrada do projeto: [README.md](../README.md).

## Camadas DDD (resumo)

```
presentation  →  application  →  domain
                    ↓
              infrastructure (ports/adapters)
```

| Camada | Módulos | Aggregate roots |
|--------|---------|-----------------|
| Application | ~132 | `Orchestrator`, `ExecutionManager` |
| Domain | ~28 | `RiskManager`, modelos `trade`, políticas AntiTrendLock |
| Infrastructure | ~49 | adaptadores Deriv, Redis, Triton, MinIO |
| Presentation | 1 | `terminal/logger` |

Regra: **domain** não importa application nem infrastructure. **Application** orquestra domain + ports; implementações concretas vêm de `infra_factory`.
