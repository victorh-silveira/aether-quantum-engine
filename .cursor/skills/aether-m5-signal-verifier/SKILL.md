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
3. **Filtro Anti-Loss M5 (ancora hibrida)**:
   - **Ancora hibrida**: ancora primaria = janela ops N=3 velas M5 fechadas; confirmacao secundaria = ultima vela micro fechada. Se ambas concordam, body = max; se discordam, body = min (fraqueza). Telemetria: `anti_loss_anchor_mode`, `anti_loss_anchor_agree`.
   - **EMA Slope (2-pontos + EMA9 rapido)**: EMA9 slope com tolerancia `slope_tol * 0.6` para deteccao precoce de reversao. EMA21 slope agora compara [-1] vs [-2] (lag 5min vs antigo 10min). Cache de EMA por ciclo evita recomputacao.
   - **RSI Momentum M5**: Vetos estritos em divergencias graves ($RSI < 0.38$ para CALL, $RSI > 0.62$ para PUT). RSI le de `micro_indicators` (forming patched = real-time).
   - **Janela Operacional (`ops_window_bars = 3`)**: Avalia o corpo liquido das ultimas 3 velas M5 (15 min acumulados), combinada com ultima vela fechada via ancora hibrida.
4. **Neg Edge Bilateral & Z-Score Panic**:
   - Trava de pânico caso o Z-score de volatilidade esteja descalibrado ($Z < -2.0$ para CALL ou $Z > +2.0$ para PUT).

