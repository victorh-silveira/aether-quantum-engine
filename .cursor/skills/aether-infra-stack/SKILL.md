---
name: aether-infra-stack
description: >-
  Sobe e verifica stack Docker Aether (Redis, Timescale, MinIO, Triton, meta).
  Use when containers unhealthy, Redis timeouts, MinIO TorchScript missing,
  Triton/meta unreachable, or the user mentions docker compose profiles.
---

# Infra stack

## Passos

1. Ler `docs/infra-docker.md` (profiles `core` / `gpu`+overlay / `cpu` / `ml`)
2. `make docker-up` (GPU), `docker-up-cpu` ou `docker-up-core`
3. Health: binds `127.0.0.1`; Redis/TS/MinIO/Triton/Meta
4. Meta: exige `.pkl` em `meta-models/` (`ready` no `/health`)
5. Triton/meta: so exigir no motor se settings `enabled`/`require_*` true
6. Nao desligar resiliencia para mascarar rede

Skill irma: `aether-settlement-debug` se a fila Redis for o sintoma.
