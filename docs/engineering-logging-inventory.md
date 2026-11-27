# Inventario de Logging Aether

Mapa SSOT das fontes de log do motor continuo Laya System 1. Contrato normativo: [`engineering-observability.md`](engineering-observability.md).

## Camadas e Fontes Atuais

| Camada | Modulo | Padrao / Eventos | Notas |
| --- | --- | --- | --- |
| Presentation | [`logger.py`](../src/presentation/terminal/logger.py) | `setup_logger` / `get_logger` | logger `AETH`, zero emojis |
| Contexto | [`log_context.py`](../src/presentation/terminal/log_context.py) | `bind_log_context` | prefixo `[cN|SYM|t=...]` |
| SSOT knobs | `config/settings.json` (`logging`) | `resolve_logging_config` | `level`, `log_file`, `quiet_channels` |
| Ciclo Laya | [`cycle_indicators_logger.py`](../src/application/services/orchestrator/cycle_indicators_logger.py) | `INDICATORS`, `LAYA_DECISION`, `MODELO` | bloco multi-linha com 2 casas decimais |
| Decisao & Skip | [`trading_cycle_entry.py`](../src/application/services/orchestrator/trading_cycle_entry.py) | `INIT`, `SKIP \| ...`, `EXEC_ERR` | `SKIP` registrado sem chamada `proposal` |
| Sizing & Ordem | [`laya_quote_sizing.py`](../src/infrastructure/handlers/laya_quote_sizing.py) e [`execution_orders.py`](../src/application/services/orchestrator/execution_orders.py) | `KELLY`, `RECOVERY`, `EXEC` | stake final revalidada na Deriv |
| Liquidacao & Stop | [`settlement_outcome.py`](../src/application/services/orchestrator/settlement_outcome.py) e [`settle_log.py`](../src/presentation/terminal/settle_log.py) | `SETTLE`, `STOP_WIN` | `STOP_WIN` disparado por `realized_session_pnl` |
| Treino GPU | `tools/laya_workflow/` | `TRAIN_*`, `CALIB_*`, `EVAL_*` | pipeline offline CUDA (`app-prepare`, `app-train`, `app-evaluate`) |
| Scripts QA | `scripts/operations/clean_workspace.py` | `print` estruturado | lint, validate, test e security |

## Nomes de Logger

| Nome | Uso |
| --- | --- |
| `AETH` | motor live (`run.py` / `engine_session` / orquestrador) |
| `AETH.ops` | operacoes auxiliares e TimescaleDB |

## Telemetria Nativa

- OpenTelemetry / Prometheus embutido (`:9100/metrics` no host, `:9090` Prometheus) via `otel_business_instrumentor.py` e `prometheus_business_format.py`.
- Dashboards Grafana provisionados (`:3000`) e traces no Tempo (`:4318`).
