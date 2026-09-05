# Infraestrutura Docker do Aether

Stack **hibrida**: motor Python 3.13 + asyncio no host (CUDA local); Redis, TimescaleDB, MinIO, meta-classifier e loss-classifier em containers. Inferencia TCN = eager/CUDA local (`data/dl`).

**Documentacao completa:** [`docs/infra-docker.md`](../../docs/infra-docker.md)  
**CloudOps sênior:** [`docs/engineering-devops-cloudops-senior.md`](../../docs/engineering-devops-cloudops-senior.md)  
**Arquitetura senior:** [`docs/engineering-architecture-senior.md`](../../docs/engineering-architecture-senior.md)

## Comandos rapidos

```bash
make docker-up          # core+ml
make docker-up-core     # so Redis/Timescale/MinIO
make docker-smoke
make sanitize-run       # DESTRUTIVO: limpa data/dl + meta/loss + data runtime
make docker-rebuild     # rebuild meta/loss e recarrega pkls (preserva TCN e meta_lgbm)
make docker-reset       # DESTRUTIVO: sanitiza + limpa volumes + bootstrap + sobe
```

Portas em `127.0.0.1`: Redis 6379, Timescale 5432, MinIO 9000/9001, Meta 8005, Loss 8006.

`aether-minio-init` e job oneshot (`restart: no`): sobe apos MinIO healthy, cria bucket `dl-models` + ILM `optuna/` ~7d e termina com **Exit 0**. Ver `Exited (0)` em `docker ps -a` e o comportamento esperado (nao e crash). Meta/loss so sobem apos `service_completed_successfully`.

## Loss-classifier (profile `ml`)

- Env: `LOSS_BOOTSTRAP_EXIT_N` **8**, `LOSS_VETO_P_LOSS_FLOOR` **0.65** (soft Kelly)
- Hard FLIP `hard_p_loss_floor` **0.90** + `flip_require_auto_learn` vivem no motor (`config/settings.json`)
- Apos mudar env: `docker compose ... up -d --force-recreate aether-loss-classifier`

Profile `ml`: `.pkl` em `meta-models/` (`train_meta_*`); loss sobe sem pkl (telemetria; veto apos `/learn`+ready_n) ou bootstrap `train_loss_classifier.py`.
