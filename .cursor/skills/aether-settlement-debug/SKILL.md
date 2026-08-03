---
name: aether-settlement-debug
description: >-
  Depura liquidacao Aether (fila Redis settlement:queue:priority, profit_table,
  orphans, tolerancia 90s). Use when logs show liquidacao estagnada, SETTLE,
  orphans, or pending contracts not resolving.
---

# Settlement debug

## Passos

1. Confirmar contract_id na fila `settlement:queue:priority`
2. Expiry + tolerancia 90 s; poll >= 2 s
3. Se estagnado: profit_table / reconcile — nao flush cego da fila
4. Orphans na janela — `SETTLE: estado reconciliado` pode ser OK
5. Apos resolve: pending/linear; limpeza `recovery:skip_counter`
6. Spam de log: manter SettlementSpamFilter

Doc: `docs/engineering-settlement.md`
