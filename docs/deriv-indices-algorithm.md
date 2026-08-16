# Indice Volatility 10 (`R_10`) — M1

Universo operacional unico: **`R_10`** (Volatility 10 / Deriv). Serie sintetica 24/7 com alvo de volatilidade ~10% anualizada. Timeframe operacional **M1** (micro/MINI **60 s**).

Referencia: [dTrader R_10 M1](https://dtrader.deriv.com/?chart_type=candle&interval=60&symbol=R_10&trade_type=rise_fall).

---

## 1. Relogio SSOT

| Item | Valor |
|------|--------|
| Simbolo API | `R_10` |
| Contrato | `RISE_FALL` **5 m** (`duration=5`, `duration_unit=m` via `ops_contract_duration_minutes`; label TCN N ∈ {15,20,…,60} eleito no treino; **SSOT atual `label_horizon_bars=55`**) |
| Micro / MINI OHLC | **60 s** (M1) |
| Macro OHLC | **7200 s** (ratio macro:micro **1:120**) |
| Ciclo / assinatura | **60 s** (alinhado ao fecho da vela M1); `exec_empty_retry` **60 s** |
| Lookback TCN | **480** barras micro (`[1, 480, 34]`; ~8 h @ 60 s) |
| Payout SSOT | **0.72** (live R_10 M1; cover = `pending/0.72`) |
| Soft Recovery | amort **1/1**, `cover_multiple` **1.50** (cover pleno) |
| Stop-win | `large_account_stop_win_pct` **3.0%** composto |
| Settle wait / tolerancia | poll **0.5 s** / tolerancia **90 s**; timeout pos-ciclo **1200 s** |
| Watchdog stale tick | **300 s** |

SSOT: `config/settings.json` + `app/src/domain/symbols/drift_symbols.py`.

---

## 2. Pipeline

- TCN em barras M1; Cal/Margin; SCALE adapta sem hard SKIP.
- Soft Kelly: `signal_skip` / loss-clf (sem flip de lado pos-LOSS).
- EXPLORE Kelly / RECOVER cover `pending/payout` (`amort` 1/1, `cover_multiple` **1.50**).

---

## 3. Migracao

1. Invalidar checkpoints de gran **180** (legado M3) e contratos **3/6/9/15 m** da grade antiga `{1,2,3,5}`.
2. Re-hidratar Timescale **60/7200**; retreinar TCN/meta/loss-clf (`launch-train` + `make docker-rebuild`).
3. Confirmar `contracts_for` autenticado: Rise/Fall **15…60 m** (passo 5) disponivel na grade N.

---

## 4. Referencias

- [engineering-orchestrator.md](engineering-orchestrator.md)
- [engineering-settings-ssot.md](engineering-settings-ssot.md)
- [binary-senior-playbook.md](binary-senior-playbook.md)
- [medallion.md](medallion.md)
- [README.md](../README.md)
