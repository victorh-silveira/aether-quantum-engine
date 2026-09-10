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
5. Meta: exige `.pkl` em `meta-models/` (`ready`+`model_loaded` no `/health`; hot-reload); `label_scale` + clamp **[-1,+0.85]**; nao promover `meta_online_*` sobre `meta_lgbm.pkl` sem gate
6. Loss: `aether-loss-classifier` — motor **FLIP** so apos auto_learn; young pe>=0.55 / mature pe>=0.58 (`veto_mode=hard`; shrink N; tape telemetria; T=2 se `n_train < LOSS_YOUNG_TEMP_N` **32**); saida do seed = `LOSS_BOOTSTRAP_EXIT_N` **12** (nao `LOSS_READY_N` **32**); `LOSS_RETRAIN_MIN_N` **12**; `veto_ready` se `n_train>=LOSS_READY_N` **32** e nao `degenerate`; seed `loss_bootstrap_live64` ou `loss_seed_real{n}`; schema_hash 24D semantico; `/v1/learn` idempotente por `contract_id`; apos schema 24D novo: `make docker-rebuild`; telemetria `[GATES] || LOSS_CLF` e `QUALITY` no settle; logs HEALTH/LEARN com `schema=` `degenerate=` `source=`
7. Meta: `aether-meta-classifier` — `META_RETRAIN_MIN_N` **32**; schema_hash 23D; fallback motor sem edge `0.0` falso (`meta_applied=false`); apos mudar env: **restart** meta/loss
8. Meta/loss: so exigir no motor se settings `enabled`/`require_*` true
9. Recarregar ML apos treino: `make docker-rebuild` (rebuild meta/loss, **nao** sanitiza `data/dl`)
10. Ciclo fresco (apaga TCN/volumes): `make docker-reset` — depois `launch-train`
11. Inferencia TCN = eager/CUDA local no host (nao ha servidor de inferencia no compose); meta/loss com timeout/fallback no motor
12. Nao desligar resiliencia para mascarar rede
13. Arquitetura / CloudOps: `docs/engineering-architecture-senior.md` + `docs/engineering-devops-cloudops-senior.md` + skills `aether-architecture-senior` / `aether-devops-cloudops`

Skill irma: `aether-settlement-debug` se a fila Redis for o sintoma; `aether-devops-cloudops` para endurecimento Compose/SQL.
