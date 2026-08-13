# Indice Volatility 10 (`R_10`) — M3

Universo operacional unico: **`R_10`** (Volatility 10 / Deriv). Serie sintetica 24/7 com alvo de volatilidade ~10% anualizada. Timeframe operacional **M3** (micro/MINI **180 s**).

Referencia: [dTrader R_10 M3](https://dtrader.deriv.com/?chart_type=candle&interval=3m&symbol=R_10&trade_type=rise_fall).

---

## 1. Relogio SSOT

| Item | Valor |
|------|--------|
| Simbolo API | `R_10` |
| Contrato | `RISE_FALL` **3 m** (`duration=3`, `duration_unit=m`) |
| Micro / MINI OHLC | **180 s** (M3) |
| Macro OHLC | **7200 s** (ratio macro:micro **1:40**) |
| Ciclo / assinatura | **180 s** (alinhado ao fecho da vela M3; contrato **3 m**); `exec_empty_retry` **180 s** |
| Lookback TCN | **480** barras micro (`[1, 480, 34]`; ~24 h @ 180 s) |
| Payout SSOT | **0.72** (live R_10 M3; cover = `pending/0.72`) |
| Soft Recovery | amort **1/1**, `cover_multiple` **1.50** (cover pleno) |
| Stop-win | `large_account_stop_win_pct` **3.0%** composto |
| Settle wait / tolerancia | poll **0.5 s** / tolerancia **90 s**; timeout pos-ciclo **1200 s** |
| Watchdog stale tick | **300 s** |

SSOT: `config/settings.json` + `app/src/domain/symbols/drift_symbols.py`.

---

## 2. Pipeline

- TCN em barras M3; Cal/Margin; SCALE adapta sem hard SKIP.
- Soft Kelly: `signal_skip` / loss-clf (sem flip de lado pos-LOSS).
- EXPLORE Kelly / RECOVER cover `pending/payout` (`amort` 1/1, `cover_multiple` **1.50**).

---

## 3. Migracao

1. Invalidar checkpoints de gran **60/120/300/900** (legado OTC_SPC M15 / M2) e contratos **2 m** / **15 m**.
2. Re-hidratar Timescale **180/7200**; retreinar TCN/meta/loss-clf.
3. Confirmar `contracts_for` autenticado: Rise/Fall **3 m** disponivel.

---

## 4. Referencias

- [engineering-orchestrator.md](engineering-orchestrator.md)
- [engineering-settings-ssot.md](engineering-settings-ssot.md)
- [binary-senior-playbook.md](binary-senior-playbook.md)
- [medallion.md](medallion.md)
- [README.md](../README.md)
