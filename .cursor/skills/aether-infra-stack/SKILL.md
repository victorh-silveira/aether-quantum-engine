---
name: aether-infra-stack
description: >-
  Sobe e verifica stack Docker Aether (Redis, Timescale, MinIO, indicador,
  Prometheus, Grafana). Use when containers unhealthy, Redis timeouts, MinIO
  TorchScript missing, or the user mentions docker compose profiles.
---

# Infra stack

## Passos

1. Ler `docs/infra-docker.md` + `docs/engineering-devops-cloudops-senior.md` (profiles `core`, `telemetry`, `indicator`)
2. `make docker-up`
3. Health: binds `127.0.0.1`; Redis(:6379)/Timescale(:5432)/MinIO(:9000)/Indicador(:9102)/Prometheus(:9090)/Grafana(:3000)
4. `make docker-logs`: default = servicos running; tail default 200
5. Indicador: subsistema isolado de analise quantitativa e confluencia institucional
6. Recarregar indicador: `make docker-rebuild`
7. Reset destrutivo (volumes): `make docker-reset`
8. Inferencia TCN = eager/CUDA local no host (nao ha servidor de inferencia no compose)
9. Nao desligar resiliencia para mascarar rede
10. Arquitetura / CloudOps: `docs/engineering-architecture-senior.md` + `docs/engineering-devops-cloudops-senior.md` + skills `aether-architecture-senior` / `aether-devops-cloudops`

Skill irma: `aether-settlement-debug` se a fila Redis for o sintoma; `aether-devops-cloudops` para endurecimento Compose/SQL.
