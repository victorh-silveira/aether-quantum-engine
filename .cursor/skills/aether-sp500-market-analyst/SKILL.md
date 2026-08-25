---
name: aether-sp500-market-analyst
description: >-
  Analisa o comportamento do mercado real S&P 500 (OTC_SPC / US 500) em timeframe M15 (900s),
  correlacionando dinamica de sessoes (NY Open, London), volatilidade ATR, medias de 15m e regime macro D1 (120 velas).
---

# SP 500 Market Analyst (M15 / D1)

Especialista em dinâmicas do mercado real S&P 500 para opções binárias e contratos direcionais na Deriv.

## Universo Operacional
- **Símbolo**: `OTC_SPC` (US 500 / S&P 500 Real Market Index OTC)
- **Micro / Operacional**: M15 (900 segundos)
- **Macro / Treinamento**: D1 (86.400 segundos), histórico de 120 velas diárias
- **Duração do Contrato**: 15 minutos fixos (`params.duration = 15`, `duration_unit = "m"`)
- **Payout Médio Mercado Real**: ~85% (`payout = 0.85`)

## Checklist Analítico M15
1. **Sessões e Horários Críticos**:
   - Abertura de NY (09:30 - 10:30 EST): Pico de volatilidade e expansão de range.
   - Fechamento da Europa / Almoço NY (11:30 - 13:00 EST): Redução de volume e potencial consolidação (chop).
   - Fechamento de NY (15:30 - 16:00 EST): Rebalanceamento institucional.
2. **Filtros Técnicos M15**:
   - **EMA 9 vs EMA 21**: Alinhamento de tendência nas velas de 15m.
   - **RSI 14 (M15)**: Sobrevenda extrema (< 38) vetando CALL se momentum for de baixa; Sobrecompra (> 62) vetando PUT se momentum for de alta.
   - **Janela de Confirmação (`ops_window_bars = 3`)**: As últimas 3 velas M15 (45 min) devem demonstrar deslocamento a favor da direção pretendida.
