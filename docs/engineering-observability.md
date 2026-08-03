# Observabilidade e logs

Presentation: `app/src/presentation/terminal/logger.py`. Dedupe: `log_dedupe.py`.

## Principios

- Logs em PT-BR, sem emoji
- Dedupe / spam filter para settlement e mensagens repetidas
- Processo > narrativa: ler `gate_reason` antes do P&L

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

- `SettlementSpamFilter` no logger de terminal reduz ruido SETTLE
- Nao “consertar” ausencia de trade removendo dedupe

Diagnostico completo: doutrina + skill `aether-session-review`.
