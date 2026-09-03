---
name: aether-v75-market-analyst
description: >-
  Analisa o comportamento do índice de volatilidade 75 (1s) (1HZ75V) em timeframe M5 (300s),
  correlacionando dinamica de volatilidade continua 24/7, ATR, medias de 5m e regime macro D1 (365 velas).
---

# Volatility 75 (1s) Market Analyst (M5 / D1)

Especialista em dinâmicas do índice sintético Volatility 75 (1s) (`1HZ75V`) para opções binárias e contratos direcionais na Deriv.

## Universo Operacional
- **Símbolo**: `1HZ75V` (Volatility 75 (1s) Index / Deriv)
- **Micro / Operacional**: M5 (300 segundos, 500 velas)
- **Macro / Treinamento**: D1 (86.400 segundos), histórico de 365 velas diárias (1 ano)
- **Duração do Contrato**: 5 minutos fixos (`params.duration = 5`, `duration_unit = "m"`)
- **Payout Médio**: ~85% a ~95% (`payout = 0.85`)

## Checklist Analítico M5
1. **Regime de Volatilidade Contínua 24/7**:
   - Sem gaps de fechamento; volatilidade uniforme em todos os horários.
   - Detecção de regimes de expansão e compressão via ATR.
2. **Filtros Técnicos M5**:
   - **EMA 9 vs EMA 21**: Alinhamento de tendência nas velas de 5m.
   - **RSI 14 (M5)**: Anti-loss momentum — CALL vetado se $\text{RSI}_{\text{M5}} < 0.35$; PUT vetado se $\text{RSI}_{\text{M5}} > 0.65$ (`why=anti_loss_rsi_momentum`).
   - **Janela de Confirmação (`ops_window_bars = 3`)**: As últimas 3 velas M5 (15 min acumulados) devem demonstrar deslocamento a favor da direção pretendida.
