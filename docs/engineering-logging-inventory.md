# Inventario de logging Aether

Mapa SSOT das fontes de log. Contrato normativo: [`engineering-observability.md`](engineering-observability.md).

## Camadas

| Camada | Volume (aprox.) | Padrao | Notas |
|--------|-----------------|--------|-------|
| Runtime `app/src` `logger.*` | ~208 calls / ~72 arquivos | `logging.getLogger("AETH")` apos `setup_logger` | **0** `print(` |
| Presentation | [`logger.py`](../app/src/presentation/terminal/logger.py) | `setup_logger` / `get_logger` | filtros Blank / Cooldown / Settlement |
| Contexto | [`log_context.py`](../app/src/presentation/terminal/log_context.py) | `bind_log_context` | prefixo `[cN|SYM]` |
| SSOT knobs | `config/settings.json` → `logging` | `resolve_logging_config` | level, log_file, quiet_channels |
| Audit ciclo | `market_audit_log.py`, `dl_cycle_log.py` | tags CLUSTER/IND/EXEC/RESOLVED | dedupe via `log_dedupe.py` |
| SETTLE | `settle_log.py` + orquestrador | `SETTLE.{canal}:` | rate-limit por canal+tick |
| Scripts treino | `train_meta_*`, `check_dl_deploy_gate` | logger `AETH.meta` / `AETH.train` | sem print no caminho critico |
| Scripts QA | `clean_workspace.py` | `print` (~22) | fora do escopo de polimento live |
| Monitor | `live_monitor.py` | logger `MONITOR` | parseia `[CLUSTER]` em `engine.log` |
| Infra meta | `infra/docker/meta-classifier/app.py` | ~18 logs | sidecar; inventario apenas |

## Hotspots runtime (poluicao historica)

| Arquivo | Papel |
|---------|-------|
| `orchestrator/ws_bootstrap.py` | AUTH / STRM / SETTLE subscribe |
| `orchestrator/settlement_queue_ops.py` | fila Redis SETTLE |
| `infrastructure/api/websocket_manager.py` | WSS (preferir DEBUG em poll) |
| `infrastructure/handlers/stream_handler.py` | DATA / STREAM sync |
| `deep_learning/dl_bootstrap_train.py` | bootstrap treino |

## Nomes de logger

| Nome | Uso |
|------|-----|
| `AETH` | motor live (`run.py` / `engine_session`) |
| `AETH.meta` | treino meta offline |
| `AETH.train` | gate deploy DL / scripts de treino pontuais |
| `AETH.ops` | ensure_timescale e ops auxiliares |
| `MONITOR` | live_monitor (arquivo dedicado) |

## Fora de escopo deste inventario operacional

- OpenTelemetry / Prometheus
- JSON puro (quebraria regex do monitor)
- Migrar `print` de `clean_workspace.py`
