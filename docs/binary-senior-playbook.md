# Playbook trader senior — binarias 120s (`R_10`)

Postura: **seletividade > atividade**. SKIP coerente e sucesso de processo.

Hierarquia: TCN Cal/Margin → indicadores (ADX/RSI/Hurst/path) → edge meta ≥ 0 → CALL/PUT → Kelly.

## Quando operar

| Lado | Condicoes tipicas |
|------|-------------------|
| CALL | Cal ≥ 0.55 (margin ≥ `hard_cal_margin_floor` 0.05), ADX ≥ 0.16, RSI/path nao contra, edge ≥ 0 |
| PUT | Idem com vies PUT |
| SKIP | Qualquer veto da tabela abaixo |

## Catalogo SKIP (`gate_reason`)

| Razao | Significado |
|-------|-------------|
| `cal_margin_floor` | Cal na zona cinza |
| `adx_min` / `adx_starvation` | Tendencia fraca / microestrutura |
| `hurst_noise` / `hurst_missing` | Regime ruído H~0.5 ou Hurst ausente |
| `rsi_trend_misalign` | `align_rsi_trend` e RSI/DI contra TCN |
| `indicator_discordance` | Votos/TA contra TCN |
| `adverse_micro_path` | Path micro adverso |
| `meta_negative_edge` | Edge meta abaixo do piso |
| `val_accuracy_gate` | ACC abaixo de 0.53 — retreinar, nao afrouxar |
| `kelly_no_edge` | Kelly sem edge |

Codigo: `execution_senior_skip.py` (`SENIOR_SKIP_REASONS`).

## Knobs SSOT (senior)

- `hard_cal_margin_floor: 0.05`, `align_rsi_trend: true`
- `indicator_gating.adx_min: 0.16`, noise Hurst `0.47–0.53`, `veto_missing_hurst: true`
- `quality_gate` margem regular `0.10`, `min_adx_threshold: 0.16`
- `min_validation_accuracy_gate: 0.53`

Ver doutrina [`llm-trading-doctrine.md`](llm-trading-doctrine.md) e [`engineering-settings-ssot.md`](engineering-settings-ssot.md).
