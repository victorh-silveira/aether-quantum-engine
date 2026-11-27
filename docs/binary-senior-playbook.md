# Playbook de Execucao Senior: Opcoes Binarias Rise/Fall M5 (`R_10`)

## 1. Principios Operacionais

1. **Autoridade de Decisao:** **100% da direcao binaria (`CALL`/`PUT`)** pertence ao **Laya System 1** (`convaiinnovations/laya`, `vegaml=False`), alimentado por velas **M5 fechadas (`300s`)**, pacote completo de indicadores causais e narrativa tecnica em 3 estados.
2. **Ativo Unico:** `R_10` (Volatility 10 Index — operacao 24/7 na Deriv).
3. **Resiliencia & Filtros de Qualidade (`SKIP`):** Antes de enviar qualquer ordem, os filtros configuraveis em `laya.skip_policy` e `risk_management.params` garantem qualidade de execucao sem alterar probabilidades do Laya.

---

## 2. Checklist de Execucao do Ciclo M5

1. **Janela Temporal (`max_entry_delay_seconds: 30`):** rejeita entradas atrasadas apos o fechamento da vela M5 (`atraso_vela`).
2. **Qualidade de Vela (`min_body_atr: 0.15`):** bloqueia velas doji/indecisao (`vela_doji`).
3. **Extremos de Banda (`bb_extreme_margin: 0.05`):** bloqueia perseguicao de topo em `CALL` (`BB%B >= 0.95`) e fundo em `PUT` (`BB%B <= 0.05`) (`extremo_banda`).
4. **Momentum e Rejeicao (`block_counter_momentum`, `max_counter_wick_atr: 0.75`):** bloqueia entradas contra vela+MACD e contra pavio de rejeicao dominante (`pavio_rejeicao`).
5. **Regime de Volatilidade (`min_atr_ratio: 0.70`, `max_atr_ratio: 1.80`):** evita mercado excessivamente comprimido ou choques extremos (`volatilidade_anomala`).
6. **Viabilidade Economica (`min_payout_rate: 0.75`):** valida payout liquido minimo na cotacao preliminar e final (`payout_baixo`), evitando degradacao de breakeven.

---

## 3. Gestao de Capital (`kelly_recovery`) e Telemetria

- **Base Kelly:** `fraction: 0.05`, `max_base_pct: 0.0125` (`1.25%` maximo em entradas base) e piso continuo de **`1.00%`** (`$1.00` minimo).
- **Recuperacao por Payout:** amortizacao da perda pendente (`pending_loss`) em etapas pelo payout real cotado, com teto absoluto de **`5.00%`** da banca por ordem (sem Martingale 2x).
- **Stop-Win Realizado:** `stop_win_pct: 0.05` (`5%`), disparado exclusivamente pelo lucro realizado da sessao (`realized_session_pnl`), com saldo sincronizado via `balance_after` real e telemetria `SESSION_STATS | W=... L=... WR=...% | PnL=... | meta=...% | pending=...` a cada liquidacao.
