---
name: aether-binary-senior
description: >-
  Avalia sessoes live no estilo trader senior de opcoes binarias M2
  (CALL/PUT/SKIP tecnico ou signal_skip 1.1; OHLC 120s) no indice
  R_10 (Volatility 10). Use when analyzing CLUSTER/Cal/Edge logs, gate_reason
  tecnico, or when the user mentions playbook senior, SKIP, Volatility, M2, ou binarias R_10.
---

# Playbook senior binario (`R_10` / M2)

Ler `docs/binary-senior-playbook.md` e `docs/deriv-indices-algorithm.md`.

Universo: **Volatility 10** (`R_10`) — **M2** (contrato **2 m**; ciclo **60 s**; micro/MINI **120 s**).

## Checklist (escopo 1.1)

1. Bloqueio tecnico? (`training`/`data`/`deploy`/`predict_error`) — senao segue TCN/SCALE
2. Catalogo `signal_skip`? mini/cal = soft Kelly — senao candidato segue (sem flip pos-LOSS)
3. ACC/deploy de treino >= 0.53 quando o tema for modelo; checar `label_call_frac` / majority-collapse
4. Cal/Margin/Edge — telemetria; Kelly usa Cal
5. SCALE adapt? majority_votes / tape / mili — sem SKIP por escala
6. Loss-clf: soft vs FLIP (`p_loss>=0.90`, `veto_ready`); seed com p_loss real (sem COLD_START)
7. Chop/neg_edge: soft Kelly (`CHOP_SOFT` / `NEG_SOFT`); sem EXEC_EMPTY de sinal
8. RECOVER vs EXPLORE; pending/cover; caps **5%**; explore piso **0.25%** (M2); Kelly `fraction` **0.08**
9. EXEC_EMPTY tecnico = sucesso de processo quando coerente; soft sinal continua EXEC

## Proibido

- Revenge sizing pos-LOSS; force_trade_every_cycle como fix de EXEC_EMPTY
- Quality gate amplo (RSI/price_zone/SIDE_EQ block) como “solucao”
- Narrar mao quente / reversao sem evidencia de log

## Referencias

- `docs/llm-trading-doctrine.md`
- `docs/binary-senior-playbook.md`
- `AGENTS.md`
