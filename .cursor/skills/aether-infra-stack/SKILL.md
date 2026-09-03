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

1. Ler `docs/infra-docker.md` (profiles `core` / `ml`)
2. `make docker-up` (core+ml) ou `docker-up-core`
3. Health: binds `127.0.0.1`; Redis/TS/MinIO/Meta(:8005)/Loss(:8006)
4. Meta: exige `.pkl` em `meta-models/` (`ready`+`model_loaded` no `/health`; hot-reload)
5. Loss: `aether-loss-classifier` — soft Kelly floor **0.65**; hard FLIP **0.90** so com `veto_ready` + `flip_require_auto_learn` **true** (seed/`auto=0` = SOFT ate sair bootstrap); `LOSS_BOOTSTRAP_EXIT_N` **16**; buffer persistido; `/learn` + retrain a cada trade; motor `LOSS_CLF || LEARN` / `SOFT` / `FLIP_BLOCK:*`
6. Apos mudar env do loss-clf: **restart** `aether-loss-classifier`
7. Meta/loss: so exigir no motor se settings `enabled`/`require_*` true
8. Recarregar ML apos treino: `make docker-rebuild` (rebuild meta/loss, **nao** sanitiza `data/dl`)
9. Ciclo fresco (apaga TCN/volumes): `make docker-reset` — depois `launch-train`
10. Inferencia TCN = eager/CUDA local no host (nao ha servidor de inferencia no compose); meta/loss com timeout/fallback no motor
11. Nao desligar resiliencia para mascarar rede
12. Arquitetura senior: `docs/engineering-architecture-senior.md` + skill `aether-architecture-senior`

Skill irma: `aether-settlement-debug` se a fila Redis for o sintoma.
