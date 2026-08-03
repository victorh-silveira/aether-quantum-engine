# Orquestrador e ciclo

Ciclo operacional do motor. Inventario de arquivos: [`structure.md`](structure.md) §orchestrator. Arquitetura: [`arquitetura.md`](arquitetura.md) §3.

## Relogio

- Fronteira / ciclo: **`signature_boundary_seconds` / `cycle_interval_seconds` = 120 s** (prefixos legados `m5` / `m15` no codigo)
- Micro OHLC: **120 s** (`data_handler.micro_granularity`)
- Macro OHLC (DL): **600 s** (`data_handler.granularity`)
- Proporcao multi-timeframe **1:5**

## Pipeline do ciclo

```text
warmup/buffer → training_gate → collect decisoes DL
  → resolve direcao/gates → rank/pick → ExecutionManager
  → settlement worker → post_settlement → proxima fronteira
```

| Etapa | Onde |
|-------|------|
| Run loop | `orchestrator_run_loop.py`, `engine_session.py` |
| Assinatura dados | `orchestrator_data_signature.py` |
| Collect | `execution_collect*.py` |
| Execucao | `execution_manager.py`, `execution_orders.py` |
| Settlement | `settlement_*.py`, `orchestrator_settlement_queue.py` |
| Pos-liquidacao | `post_settlement_*.py` |
| Persistencia | `orchestrator_persistence.py`, `orchestrator_atomic_state.py` |
| Watchdog | `watchdog_service.py` |

## Gates de fase

- **FASE TREINO:** sem ordens ate modelos da sessao prontos
- **FASE OPERACAO:** `mandatory_trade_each_cycle: true`, `force_trade_every_cycle: false`
- Lock/barreira serializa inferencia, liquidacao e persistencia

## Diagnostico rapido

| Sintoma | Ver |
|---------|-----|
| Ciclo nao dispara | signature, warmup buffer, idle watchdog |
| So EXEC_EMPTY | gates (Cal/Edge/ACC) — processo pode estar correto |
| Travado apos trade | settlement queue / post_settlement |
| Reconnect loop | watchdog stale + cooldown |

Skill: `aether-cycle-debug`.
