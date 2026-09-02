---
name: aether-m5-signal-verifier
description: >-
  Verifica e valida a esteira completa de sinais tecnicos, filtros anti-loss, fusao EV,
  EMA slope e momentum RSI calibrados para o timeframe M5 (300s).
---

# M5 Signal & Indicator Verifier

Especialista na depuração de sinais em barras de 5 minutos e validação de gates de execução.

## Pipeline de Decisão M5
1. **Leitura de Barra M5**: O ciclo do motor ocorre sincronizado ao fechamento de cada barra de 5 minutos (`cycle_interval_seconds = 120`, `signature_boundary_seconds = 300`).
2. **Fusão EV**: Pondera votos direcionais com pesos calibrados para M5.
3. **Filtro Anti-Loss M5**:
   - **EMA Slope**: Vela M5 deve confirmar direção com respeito à inclinação de EMA 9 e EMA 21.
   - **RSI Momentum M5**: Vetos estritos em divergências graves ($RSI < 0.38$ para CALL, $RSI > 0.62$ para PUT).
   - **Janela Operacional (`ops_window_bars = 3`)**: Avalia o corpo líquido das últimas 3 velas M5 (15 min acumulados).
4. **Neg Edge Bilateral & Z-Score Panic**:
   - Trava de pânico caso o Z-score de volatilidade esteja descalibrado ($Z < -2.0$ para CALL ou $Z > +2.0$ para PUT).

