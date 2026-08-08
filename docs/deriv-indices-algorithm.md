# Indice Volatility 10 (`R_10`) — M2

Universo operacional unico: **`R_10`** (Volatility 10 / Deriv). Serie sintetica 24/7 com alvo de volatilidade ~10% anualizada. Timeframe operacional **M2** (micro/MINI **120 s**).

Referencia: [dTrader R_10 M2](https://dtrader.deriv.com/?chart_type=candle&interval=2m&symbol=R_10&trade_type=rise_fall).

---

## 1. Relogio SSOT

| Item | Valor |
|------|--------|
| Simbolo API | `R_10` |
| Contrato | `RISE_FALL` **2 m** (`duration=2`, `duration_unit=m`) |
| Micro / MINI OHLC | **120 s** (M2) |
| Macro OHLC | **3600 s** (1:30) |
| Ciclo / assinatura | **60 s** (entrada a cada 1 m; contrato permanece **2 m**) |
| Lookback TCN | **720** barras micro |
| Payout SSOT | **0.72** (live R_10 M2; cover = `pending/0.72`) |
| Settle wait / tolerancia | poll **0.5 s** / **300 s**; timeout pos-ciclo **1200 s** |
| Watchdog stale tick | **600 s** |

SSOT: `config/settings.json` + `app/src/domain/symbols/drift_symbols.py`.

---

## 2. Pipeline

- TCN em barras M2; Cal/Margin; SCALE adapta sem hard SKIP.
- Soft Kelly: `signal_skip` / loss-clf (sem flip de lado pos-LOSS).
- EXPLORE Kelly / RECOVER cover `pending/payout` (`amort` 1/1).

---

## 3. Migracao

1. Invalidar checkpoints de gran **60/300/900** (legado OTC_SPC M15) e contratos **15 m**.
2. Re-hidratar Timescale **120/3600**; retreinar TCN/meta/loss-clf.
3. Confirmar `contracts_for` autenticado: Rise/Fall **2 m** disponivel.

---

## 4. Referencias

- [engineering-orchestrator.md](engineering-orchestrator.md)
- [engineering-settings-ssot.md](engineering-settings-ssot.md)
- [binary-senior-playbook.md](binary-senior-playbook.md)
- [medallion.md](medallion.md)
- [README.md](../README.md)
