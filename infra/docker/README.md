# Infraestrutura Docker do Aether

Stack **hibrida**: motor no host; Redis, TimescaleDB, MinIO, Triton, meta-classifier e loss-classifier em containers.

**Documentacao completa:** [`docs/infra-docker.md`](../../docs/infra-docker.md)

## Comandos rapidos

```bash
make docker-up          # core+gpu+ml + overlay NVIDIA
make docker-up-cpu      # core+cpu+ml sem NVIDIA
make docker-up-core     # so Redis/Timescale/MinIO
make docker-smoke
make docker-rebuild     # limpa loss-models + bootstrap cold-start + rebuild meta/loss + up
make docker-reset       # DESTRUTIVO: limpa loss-models + volumes + bootstrap + sobe
```

Portas em `127.0.0.1`: Redis 6379, Timescale 5432, MinIO 9000/9001, Triton 8000/8001, Meta 8005, Loss 8006.

GPU e CPU sao mutuamente exclusivos na mesma porta Triton. Profile `ml`: `.pkl` em `meta-models/` (`train_meta_*`); loss sobe sem pkl (telemetria; veto apos `/learn`+ready_n) ou bootstrap `train_loss_classifier.py`.
