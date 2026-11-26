# Estrutura do Repositorio

Organizacao canonica do Aether Quantum Engine na branch `feat/laya-continuous-r10`:

- `run.py`: ponto de entrada unico (`make app-run`).
- `config/settings.json`: configuracao SSOT (`R_10` M5, Laya, risco `[1%, 5%]`, infra).
- `src/domain/`: nucleo puro DDD (`indicators/`, `models/`, `risk/`, `math/`, `symbols/`, `exceptions.py`).
- `src/application/`: portas (`ports/outbound.py`) e servicos de orquestracao (`services/orchestrator/`, `auth_manager.py`, `quant_metrics_collector.py`).
- `src/infrastructure/`: adapters Deriv (`api/`, `handlers/`, `websocket/`), inferencia Laya (`inference/laya_decision_adapter.py`), estado (`state/`), persistencia (`market/`, `storage/`), telemetria (`telemetry/`) e runtime (`native/`, `serialization/`).
- `src/presentation/terminal/`: loggers e formatadores sem emojis.
- `scripts/operations/`: automacoes de QA (`clean_workspace.py`, `qa/`) e manutencao.
- `infra/docker/`: `docker-compose.yml` com perfis `core` (Redis, TimescaleDB, MinIO) e `telemetry` (Prometheus, Grafana, Tempo).
- `tests/`: suite unitaria e de integracao com meta de 100% de cobertura em `src/`.
- `docs/` e `.agents/`: documentacao tecnica e harness de agentes (rules e skills).
