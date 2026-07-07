# Documentação

| Documento | Conteúdo |
|-----------|----------|
| [structure.md](structure.md) | Layout do repositório (`app/`, `config/`, `linters/`, `infra/docker/`) e mapa de módulos |
| [arquitetura.md](arquitetura.md) | Fluxo ao vivo: WebSocket, watchdog, barreira atômica `asyncio.Lock`, DL, Triton gRPC, meta-regressor 39D, edge contínuo D-SQUEEZE, execução, risco financeiro, stop win fast-path, settlement e persistência |
| [medallion.md](medallion.md) | Filosofia quantitativa e perfil de qualidade (scoring, gate, ranking, recovery, Kelly, regressão de payoff D-SQUEEZE, meta 1% por sessão) |
| [deriv-indices-algorithm.md](deriv-indices-algorithm.md) | Algoritmo dos índices sintéticos da Deriv e estratégia do motor |
| [infra-docker.md](infra-docker.md) | Stack Docker (Redis, TimescaleDB, MinIO, Triton, meta-regressor 8005), healthchecks Python, pipeline Redis, `StateManager` |
| [deriv-api.md](deriv-api.md) | Referência Deriv + integração usada pelo motor (PAT, OTP, manutenção do broker) |
| [CHANGELOG.md](CHANGELOG.md) | Histórico de versões |

Ponto de entrada do projeto: [README.md](../README.md).
