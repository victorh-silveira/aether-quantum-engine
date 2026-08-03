# Playbook trader senior — binarias 120s (`R_10`)

Postura operacional (mandato escopo 1): **pipeline sem vetos de sinal/qualidade**. TCN resolve CALL/PUT; meta/edge/indicadores sao telemetria. SKIP apenas por bloqueio tecnico.

Hierarquia: TCN Cal/Margin (telemetria) → CALL/PUT → Kelly/caps.

## Quando operar

| Lado | Condicoes tipicas |
|------|-------------------|
| CALL | TCN resolve CALL e nao ha bloqueio tecnico |
| PUT | TCN resolve PUT e nao ha bloqueio tecnico |
| SKIP tecnico | `training` / `data` / `deploy` / `predict_error`, warm-up, stop-win, broker |

## Catalogo SKIP tecnico

| Razao | Significado |
|-------|-------------|
| `training` | Treino de sessao em andamento |
| `data` | Buffer/historico insuficiente |
| `deploy` | Checkpoint sem `deploy_ok` |
| `predict_error` | Falha de inferencia |
| Kelly `EXEC_PAUSE` | `kelly_no_edge` / stake 0 (sizing, nao veto de direcao) |

Vetos de sinal removidos do codigo: Hurst/ADX/RSI/discordance/adverse path/price zone, quality gate (cal floor, margin, meta edge, starvation), SIDE_EQ bloqueante, senior skip catalog.

## Knobs SSOT restantes (senior)

- `force_trade_every_cycle: false` (proibido como “fix”)
- `min_validation_accuracy_gate: 0.53` (treino/deploy)
- Caps Kelly / `max_safe_stake_*`

Ver doutrina [`llm-trading-doctrine.md`](llm-trading-doctrine.md) e [`engineering-settings-ssot.md`](engineering-settings-ssot.md).
