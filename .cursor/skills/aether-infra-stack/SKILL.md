---
name: aether-infra-stack
description: >-
  Sobe e verifica stack Docker Aether (Redis, Timescale, MinIO, meta,
  loss-classifier). Use when containers unhealthy, Redis timeouts, MinIO
  TorchScript missing, meta/loss unreachable, or the user mentions docker
  compose profiles.
---

# Infra stack

## Passos

1. Ler `docs/infra-docker.md` + `docs/engineering-devops-cloudops-senior.md` (profiles `core` / `ml`)
2. `make docker-up` (core+ml) ou `docker-up-core`
3. Health: binds `127.0.0.1`; Redis/TS/MinIO/Meta(:8005)/Loss(:8006); `minio-init` cria `dl-models` e sai Exit 0 (oneshot; `Exited (0)` = OK)
4. `make docker-logs`: default = servicos running (sem `minio-init`); `DOCKER_SERVICE=minio-init` para oneshot; tail default 200
5. Meta: exige `.pkl` em `meta-models/` (`ready`+`model_loaded` no `/health`; hot-reload)
6. Loss: `aether-loss-classifier` — soft Kelly floor **0.65**; hard FLIP **0.90** so com `veto_ready` + `flip_require_auto_learn` **true** (seed/`auto=0` = SOFT ate sair bootstrap); `LOSS_BOOTSTRAP_EXIT_N` **8**; buffer persistido; `/learn` + retrain a cada trade; motor `LOSS_CLF || LEARN` / `SOFT` / `FLIP_BLOCK:*`
7. Apos mudar env do loss-clf: **restart** `aether-loss-classifier`
8. Meta/loss: so exigir no motor se settings `enabled`/`require_*` true
9. Recarregar ML apos treino: `make docker-rebuild` (rebuild meta/loss, **nao** sanitiza `data/dl`)
10. Ciclo fresco (apaga TCN/volumes): `make docker-reset` — depois `launch-train`
11. Inferencia TCN = eager/CUDA local no host (nao ha servidor de inferencia no compose); meta/loss com timeout/fallback no motor
12. Nao desligar resiliencia para mascarar rede
13. Arquitetura / CloudOps: `docs/engineering-architecture-senior.md` + `docs/engineering-devops-cloudops-senior.md` + skills `aether-architecture-senior` / `aether-devops-cloudops`

Skill irma: `aether-settlement-debug` se a fila Redis for o sintoma; `aether-devops-cloudops` para endurecimento Compose/SQL.
