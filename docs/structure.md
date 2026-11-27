# Estrutura do Repositorio

Organizacao canonica do Aether Quantum Engine na branch `feat/laya-continuous-r10`:

- `run.py`: ponto de entrada unico (`make app-run`), com deteccao limpa de encerramento gracioso (`STOP_WIN` -> `exit 0`).
- `config/settings.json`: configuracao SSOT (`R_10` M5, `laya.skip_policy`, `risk_management.kelly`, `stop_win_pct`, `infra`).
- `src/domain/`: nucleo puro DDD (`indicators/`, `models/`, `risk/`, `services/laya_skip_policy.py`, `symbols/`, `config_knobs.py`).
- `src/application/`: portas (`ports/outbound.py`) e servicos de orquestracao (`services/orchestrator/`, `auth_manager.py`, `quant_metrics_collector.py`, `polars_runtime.py`).
- `src/infrastructure/`: adapters Deriv (`api/`, `handlers/`), inferencia Laya (`inference/laya_decision_adapter.py`, `laya_narrative.py`, `laya_context.py`, `laya_checkpoint.py`), estado (`state/`), persistencia (`market/`, `storage/`), telemetria (`telemetry/`) e runtime (`native/`, `serialization/`).
- `src/presentation/terminal/`: loggers e formatadores sem emojis.
- `tools/`: pipeline GPU offline (`tools/laya_workflow/`), checagem CUDA (`tools/laya_cuda_check.py`) e probe de inferencia (`tools/laya_inference_probe.py`).
- `scripts/operations/`: automacoes de QA (`clean_workspace.py`, `qa/`) e manutencao.
- `infra/docker/`: `docker-compose.yml` com perfis `core` (Redis, TimescaleDB, MinIO) e `telemetry` (Prometheus, Grafana, Tempo).
- `tests/`: suite unitaria, de integracao e gates arquiteturais (`tests/unit/architecture/test_ddd_gates.py`) com **100% de cobertura** em `src/`.
- `docs/` e `.agents/`: documentacao tecnica e harness de agentes (rules e skills).
