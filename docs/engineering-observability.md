# Observabilidade e logs

Presentation: `app/src/presentation/terminal/logger.py`.
Contexto: `log_context.py`. SETTLE: `settle_log.py`. SSOT: `logging_config.resolve_logging_config`.
Dedupe: `log_dedupe.py`. Inventario: [`engineering-logging-inventory.md`](engineering-logging-inventory.md).

## Principios

- Logs em PT-BR, sem emoji
- Dedupe / spam filter para settlement e mensagens repetidas
- Processo > narrativa: ler `gate_reason` antes do P&L
- Logger unico do motor: `AETH` via `setup_logger` / `get_logger` (idempotente)
- Treino: `AETH.meta` / `AETH.train` — sem `print` no caminho critico
- Rich no terminal nao deve bloquear o event loop; daemon/CI: sem ANSI / logs estruturados (ver [`engineering-architecture-senior.md`](engineering-architecture-senior.md) §9)

## Knobs SSOT (`logging`)

| Chave | Default | Papel |
|-------|---------|-------|
| `level` | `INFO` | Nivel do logger AETH (`DEBUG` so para diagnostico) |
| `log_file` | `logs/engine.log` | Persistencia |
| `quiet_channels` | settle_enqueue, settle_process, settle_tolerance, settle_read, ws_ping, warmup_poll, execution_flow | Canal → DEBUG via `log_settle` |

## Contrato de tags

| Tag | Nivel tipico | Frequencia | Consumidor |
|-----|--------------|------------|------------|
| CLUSTER / IND / KELLY / EXEC* / RESOLVED | INFO | ≤1/ciclo (dedupe) | session-review, live_monitor |
| SETTLE.{canal} | INFO em estado; DEBUG se quiet | rate-limit canal+tick | settlement-debug |
| WSS / AUTH / MINIO | INFO no boot; DEBUG em reconexao (exceto AVISO/ERRO) | evento | deriv-connect / infra |
| RECOV (restaurado / ciclo liberado) | INFO | reconexao | cycle-debug |
| CICLO pos-liq / SRE / RECONCILE portfolio | DEBUG | rotina settle | settlement-debug |
| EXECUTION_FLOW / WARMUP | INFO se mudou; quiet → DEBUG | dedupe | cycle-debug |

Prefixo opcional de correlacao: `[cN|SYM]` (nao quebra regex `[CLUSTER]` do monitor).

## Tags tipicas do ciclo

| Tag | Significado |
|-----|-------------|
| `MINIO` / `TorchScript` | artefatos de modelo |
| `WSS` / `AUTH` | conexao e conta |
| `SESSAO INICIADA` | banca, stop-win |
| `DATA` / `CFG` | buffer e knobs efetivos |
| `DL` | device / inferencia |
| `CLUSTER` | Prob / Cal / Margin / Edge |
| `SIDE_EQ` / `META_VETO` | equilibrio lateral / veto meta |
| `IND` | indicadores de contexto |
| `KELLY` | p, live_wr, f*, mode |
| `EXEC` / `EXEC_EMPTY` / `EXEC_PAUSE` | ordem ou veto |
| `RESOLVED` / `RISK` | resultado e pending |
| `SETTLE` / `CICLO` / `SRE` | liquidacao e limpeza |

## Filtros

- `BlankLineSquasher` — linhas em branco consecutivas
- `CooldownDeduplicationFilter` — CICLO cooling-down / resfriamento (1×/tick)
- `SettlementSpamFilter` — SETTLE/WARMUP/EXECUTION_FLOW por **canal+tick**
- `log_dedupe` — responsabilidade de conteudo (quality/EXEC_EMPTY); Filter = anti-rajada

Nao “consertar” ausencia de trade removendo dedupe.

Diagnostico: doutrina + skill `aether-session-review`.
