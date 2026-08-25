---
name: aether-m15-signal-verifier
description: >-
  Verifica e valida a esteira completa de sinais tecnicos, filtros anti-loss, fusao EV,
  EMA slope e momentum RSI calibrados para o timeframe M15 (900s).
---

# M15 Signal & Indicator Verifier

Especialista na depuração de sinais em barras de 15 minutos e validação de gates de execução.

## Pipeline de Decisão M15
1. **Leitura de Barra M15**: O ciclo do motor ocorre sincronizado ao fechamento de cada barra de 15 minutos (`cycle_interval_seconds = 900`, `signature_boundary_seconds = 900`).
2. **Fusão EV**: Pondera votos direcionais com pesos calibrados para M15.
3. **Filtro Anti-Loss M15**:
   - **EMA Slope**: Vela M15 deve confirmar direção com respeito à inclinação de EMA 9 e EMA 21.
   - **RSI Momentum M15**: Vetos estritos em divergências graves ($RSI < 0.38$ para CALL, $RSI > 0.62$ para PUT).
   - **Janela Operacional (`ops_window_bars = 3`)**: Avalia o corpo líquido das últimas 3 velas M15 (45 min acumulados).
4. **Neg Edge Bilateral & Z-Score Panic**:
   - Trava de pânico caso o Z-score de volatilidade esteja descalibrado ($Z < -2.0$ para CALL ou $Z > +2.0$ para PUT).
