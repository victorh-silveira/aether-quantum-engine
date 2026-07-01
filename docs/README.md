# Documentação

| Documento | Conteúdo |
|-----------|----------|
| [structure.md](structure.md) | Layout do repositório (`app/`, `config/`, `linters/`, `infra/docker/`) |
| [arquitetura.md](arquitetura.md) | Fluxo ao vivo: WebSocket, watchdog, DL, Triton gRPC (timeout/fallback), direção, qualidade, execução, risco financeiro, stop win por sessão ativa, settlement |
| [medallion.md](medallion.md) | Filosofia quantitativa e perfil de qualidade (scoring, gate, ranking, recovery, Kelly, meta 1% por sessão) |
| [deriv-indices-algorithm.md](deriv-indices-algorithm.md) | Algoritmo dos índices sintéticos da Deriv e estratégia do motor |
| [infra-docker.md](infra-docker.md) | Stack Docker (Redis, TimescaleDB, MinIO, Triton), sanity estressado, pipeline Redis |
| [deriv-api.md](deriv-api.md) | Referência Deriv + integração usada pelo motor |
| [CHANGELOG.md](CHANGELOG.md) | Histórico de versões |

Ponto de entrada do projeto: [README.md](../README.md).
