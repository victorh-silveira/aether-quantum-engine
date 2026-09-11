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

- Env: `LOSS_READY_N` **32** (`veto_ready`/maduro), `LOSS_RETRAIN_MIN_N` **12**, `LOSS_BOOTSTRAP_EXIT_N` **4** (saida do seed / auto_learn; nao `READY_N`), `LOSS_MIN_WIN_FOR_LOSS_RETRAIN` **4**, `LOSS_VETO_P_LOSS_FLOOR` **0.58** (telemetria no sidecar; motor ignora para SKIP)
- **FLIP** por `p_eff` (so apos auto_learn; young pe>=**0.55**; mature pe>=**0.58**; shrink N; tape telemetria; `LOSS_YOUNG_TEMP_N` **32** → T=2) vive no motor (`config/settings.json`); sem HARD SKIP/Soft Kelly
- Apos mudar env: `docker compose ... up -d --force-recreate aether-loss-classifier`

Profile `ml`: `.pkl` em `meta-models/` (`train_meta_*`); loss sobe sem pkl (telemetria; veto apos `/learn`+ready_n) ou bootstrap `train_loss_classifier.py`.
