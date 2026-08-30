# Volatility 75 (1s) Index (`1HZ75V`) — M15

Universo operacional unico: **`1HZ75V`** (Volatility 75 (1s) Index / Deriv). Ativo sintético contínuo 24/7 com contrato direcional RISE/FALL em M15. Timeframe operacional **M15** (micro/MINI **900 s**), com treinamento em **D1 (86400 s)** em **100 velas diarias**.

---

## 1. Relogio SSOT

| Item | Valor |
|------|--------|
| Simbolo API | `1HZ75V` |
| Contrato | `RISE_FALL` **15 m** (`duration=15`, `duration_unit=m`, `label_horizon_bars=1`) |
| Micro / MINI OHLC | **900 s** (M15, 500 velas) |
| Macro OHLC | **86400 s** (D1 / 365 velas de treino - 1 ano) |
| Ciclo / assinatura | **900 s** (alinhado ao fecho da vela M15); `exec_empty_retry` **900 s** |
| Lookback TCN | **30** barras |
| Payout SSOT | **0.85** (85% payout base) |
| Soft Recovery | amort **1/1**, `cover_multiple` **1.50** (cover pleno) |
| Stop-win | `compounding_rate_daily` **1.0%** / Single-Strike 1 trade |
| Settle wait / tolerancia | poll **0.5 s** / tolerancia **600 s**; timeout pos-ciclo **1200 s** |
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
