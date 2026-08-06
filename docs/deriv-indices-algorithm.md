# Indice OTC SP 500 (`OTC_SPC`) — somente M15

Universo operacional unico: **`OTC_SPC`** (S&P 500 OTC). Preco de mercado real (horario de sessao). Timeframe operacional **somente M15**.

Referencia: [dTrader OTC_SPC M15](https://dtrader.deriv.com/?chart_type=candle&interval=15m&symbol=OTC_SPC&trade_type=rise_fall).

---

## 1. Relogio SSOT

| Item | Valor |
|------|--------|
| Simbolo API | `OTC_SPC` |
| Contrato | `RISE_FALL` **15 m** (`duration=15`, `duration_unit=m`) |
| Micro / MINI OHLC | **900 s** (M15) |
| Macro OHLC | **3600 s** (1:5) |
| Ciclo / assinatura | **15 s** (entrada continua; contrato permanece **15 m**) |
| Lookback TCN | **720** barras micro (~7,5 dias) |
| Payout SSOT | **0.72** (live OTC_SPC M15; cover = `pending/0.72`) |
| Settle wait / tolerancia | poll **0.5 s** / **300 s**; timeout pos-ciclo **1200 s** |
| Watchdog stale tick | **600 s** (OTC pode ficar quieto fora do horario US) |

SSOT: `config/settings.json` + `app/src/domain/symbols/drift_symbols.py`.

---

## 2. Pipeline

- TCN em barras M15; Cal/Margin; SCALE adapta sem hard SKIP.
- Soft Kelly: `signal_skip` / loss-clf (sem flip de lado pos-LOSS).
- EXPLORE Kelly / RECOVER cover `pending/payout` (`amort` 1/1).

---

## 3. Migracao

1. Invalidar checkpoints de gran **60/300** e contratos **30 s**.
2. Re-hidratar Timescale **900/3600**; retreinar TCN/meta/loss-clf.
3. Confirmar `contracts_for` autenticado: Rise/Fall **15 m** disponivel no horario de mercado.

---

## 4. Referencias

- [engineering-orchestrator.md](engineering-orchestrator.md)
- [engineering-settings-ssot.md](engineering-settings-ssot.md)
- [binary-senior-playbook.md](binary-senior-playbook.md)
- [medallion.md](medallion.md)
- [README.md](../README.md)
