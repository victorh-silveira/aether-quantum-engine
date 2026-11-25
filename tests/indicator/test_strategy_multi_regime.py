"""Testes unitarios completos (100% cobertura) para strategy_multi_regime v2 e paridade de treino."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.domain.models.indicator_consensus import approved_signal_from_row
from src.indicator.domain import Candle
from src.indicator.pullback import evaluate, version_for_variant
from src.indicator.pullback_training import _causal_peer_slice, diagnose_all
from src.indicator.strategy_multi_regime import (
    _candle_dir,
    _eval_climax_global,
    _eval_di5_pullback,
    _eval_q2_cci14,
    _is_marubozu,
    _is_ny_premarket_entry,
    _peer_snapshot,
    _wilder_atr,
    calc_wilder_di_diff,
    evaluate_multi_regime_hf,
)


def _epoch(hour: int, minute: int = 0) -> int:
    return int(datetime(2026, 11, 18, hour, minute, tzinfo=UTC).timestamp())


def _compressed_bars(end_hour: int = 10, end_minute: int = 0, *, bull_last: bool = True) -> list[Candle]:
    end_ep = _epoch(end_hour, end_minute)
    bars: list[Candle] = []
    for idx in range(55):
        ep = end_ep - (54 - idx) * 900
        if idx < 40:
            bars.append(Candle(ep, 100.0, 105.0, 95.0, 101.0))
        elif idx < 54:
            bars.append(Candle(ep, 100.0, 101.0, 99.0, 100.2))
        elif bull_last:
            bars.append(Candle(ep, 100.0, 112.0, 99.0, 109.0))
        else:
            bars.append(Candle(ep, 100.0, 101.0, 88.0, 91.0))
    return bars


def test_helpers_and_edge_cases():
    assert version_for_variant("multi_regime_hf").startswith("pullback-1-")
    assert _wilder_atr([], 14) == 0.0
    assert _wilder_atr([Candle(0, 1, 2, 0.5, 1.5)], 14) == 0.0
    assert calc_wilder_di_diff([Candle(i * 900, 100, 101, 99, 100) for i in range(5)], 5) == 0.0
    flat = [Candle(i * 900, 100.0, 100.0, 100.0, 100.0) for i in range(20)]
    assert calc_wilder_di_diff(flat, 5) == 0.0
    assert _candle_dir(Candle(0, 10, 11, 9, 10.5)) == 1
    assert _candle_dir(Candle(0, 10, 11, 9, 9.5)) == -1
    assert _candle_dir(Candle(0, 10, 11, 9, 10.0)) == 0
    assert _is_ny_premarket_entry(_epoch(10, 30)) is True
    assert _is_ny_premarket_entry(_epoch(11, 30)) is True
    assert _is_ny_premarket_entry(_epoch(10, 0)) is False
    assert _is_marubozu(Candle(0, 100.0, 105.0, 100.0, 104.8)) is True
    assert _is_marubozu(Candle(0, 100.0, 100.0, 100.0, 100.0)) is True
    assert _is_marubozu(Candle(0, 100.0, 105.0, 98.0, 104.0)) is False
    assert _peer_snapshot("OTC_GDAXI", 900, None) == ([], [], [])
    assert _peer_snapshot("OTC_GDAXI", 900, {"OTC_GDAXI": [], "OTC_FCHI": [Candle(1800, 1, 2, 0.5, 1.5)]}) == (
        [],
        [],
        [],
    )
    bars_short = [Candle(900, 100.0, 101.0, 99.0, 100.5)]
    glb, blk, ratios = _peer_snapshot("OTC_GDAXI", 900, {"OTC_GDAXI": bars_short, "OTC_SPC": bars_short})
    assert glb == [1, 1]
    assert blk == [1]
    assert ratios == []


def test_climax_global_call_put_and_guards():
    bars_up = _compressed_bars(10, 0, bull_last=True)
    atr14, atr50 = _wilder_atr(bars_up, 14), _wilder_atr(bars_up, 50)
    assert _eval_climax_global(bars_up, atr14, atr50, [1, 1, 1, -1], [0.5, 0.6]) == "PUT"

    bars_dn = _compressed_bars(10, 0, bull_last=False)
    assert _eval_climax_global(bars_dn, atr14, atr50, [-1, -1, -1, 1], []) == "CALL"

    # Guardas: hora 20 UTC, pre-mercado NY, marubozu, doji, tr_atr <= 1.5, blk_mean_atrr >= 0.95, <3 pares, consenso < 0.65
    assert _eval_climax_global(_compressed_bars(20, 0, bull_last=True), atr14, atr50, [1, 1, 1], [0.5]) is None
    assert _eval_climax_global(_compressed_bars(11, 30, bull_last=True), atr14, atr50, [1, 1, 1], [0.5]) is None
    bars_maru = list(bars_up)
    bars_maru[-1] = Candle(bars_up[-1].epoch, 100.0, 109.2, 99.9, 109.0)
    assert _eval_climax_global(bars_maru, atr14, atr50, [1, 1, 1], [0.5]) is None
    bars_doji = list(bars_up)
    bars_doji[-1] = Candle(bars_up[-1].epoch, 100.0, 105.0, 95.0, 100.0)
    assert _eval_climax_global(bars_doji, atr14, atr50, [1, 1, 1], [0.5]) is None
    bars_small = list(bars_up)
    bars_small[-1] = Candle(bars_up[-1].epoch, 100.0, 101.0, 99.5, 100.5)
    assert _eval_climax_global(bars_small, atr14, atr50, [1, 1, 1], [0.5]) is None
    assert _eval_climax_global(bars_up, atr14, atr50, [1, 1, 1], [1.05]) is None
    assert _eval_climax_global(bars_up, atr14, atr50, [1, 1], [0.5]) is None
    assert _eval_climax_global(bars_up, atr14, atr50, [1, -1, -1, 1], [0.5]) is None


def test_q2_cci14_call_put_and_guards():
    bars_dn = _compressed_bars(9, 30, bull_last=False)
    atr14, atr50 = _wilder_atr(bars_dn, 14), _wilder_atr(bars_dn, 50)
    assert _eval_q2_cci14(bars_dn, atr14, atr50, [-1, -1, 1]) == "CALL"

    bars_up = _compressed_bars(9, 30, bull_last=True)
    assert _eval_q2_cci14(bars_up, atr14, atr50, [1, 1, -1]) == "PUT"

    # Guardas: hora 20, pre-mercado NY, marubozu, minuto != 30, atr_ratio >= 1.0, <2 pares, consenso baixo, CCI neutro
    assert _eval_q2_cci14(_compressed_bars(20, 30, bull_last=False), atr14, atr50, [-1, -1]) is None
    assert _eval_q2_cci14(_compressed_bars(11, 30, bull_last=False), atr14, atr50, [-1, -1]) is None
    assert _eval_q2_cci14(_compressed_bars(9, 15, bull_last=False), atr14, atr50, [-1, -1]) is None
    assert _eval_q2_cci14(bars_dn, 10.0, 8.0, [-1, -1]) is None
    assert _eval_q2_cci14(bars_dn, atr14, atr50, [-1]) is None
    assert _eval_q2_cci14(bars_dn, atr14, atr50, [1, 1]) is None
    assert _eval_q2_cci14(bars_up, atr14, atr50, [-1, -1]) is None
    bars_neutral = list(bars_up)
    bars_neutral[-1] = Candle(bars_up[-1].epoch, 100.0, 101.0, 99.0, 100.2)
    assert _eval_q2_cci14(bars_neutral, atr14, atr50, [1, 1]) is None


def test_di5_pullback_call_put_and_guards():
    end_ep = _epoch(13, 15)
    bull: list[Candle] = []
    price = 100.0
    for idx in range(60):
        ep = end_ep - (59 - idx) * 900
        if idx < 59:
            bull.append(Candle(ep, price, price + 2.5, price - 0.1, price + 2.0))
            price += 2.0
        else:
            bull.append(Candle(ep, price, price + 0.1, price - 0.4, price - 0.3))
    assert _eval_di5_pullback(bull) == "CALL"

    bear: list[Candle] = []
    price = 200.0
    for idx in range(60):
        ep = end_ep - (59 - idx) * 900
        if idx < 59:
            bear.append(Candle(ep, price, price + 0.1, price - 2.5, price - 2.0))
            price -= 2.0
        else:
            bear.append(Candle(ep, price, price + 0.4, price - 0.1, price + 0.3))
    assert _eval_di5_pullback(bear) == "PUT"

    last_o = bull[-1].open
    doji = list(bull)
    doji[-1] = Candle(bull[-1].epoch, last_o, last_o + 0.2, last_o - 0.2, last_o)
    assert _eval_di5_pullback(doji) is None
    no_pb = list(bull)
    no_pb[-1] = Candle(bull[-1].epoch, last_o, last_o + 2.0, last_o - 0.1, last_o + 1.5)
    assert _eval_di5_pullback(no_pb) is None


def test_evaluate_multi_regime_hf_integration_and_conflict(monkeypatch):
    bars_up = _compressed_bars(10, 0, bull_last=True)
    peers = {
        "OTC_GDAXI": bars_up,
        "OTC_FCHI": bars_up,
        "OTC_FTSE": bars_up,
        "OTC_SX5E": bars_up,
    }
    res = evaluate(
        bars_up,
        900,
        bars_up[-1].epoch + 900,
        variant="multi_regime_hf",
        allow_session_gaps=True,
        symbol="OTC_GDAXI",
        peer_bars=peers,
    )
    assert res.side == "PUT"
    assert res.reason == "setup_confirmado"
    approved = approved_signal_from_row(
        {
            "symbol": "OTC_GDAXI",
            "period_seconds": 900,
            "candle_epoch": bars_up[-1].epoch,
            "model_version": res.version,
            "side": res.side,
            "probability": None,
            "reason": res.reason,
            "model_active": True,
        },
        expected_epoch=bars_up[-1].epoch,
    )
    assert approved is not None
    assert approved.side == "PUT"
    assert approved.model_version == res.version

    # Guardas: poucas velas, virada :45, ATR invalido, sem sinal, conflito
    assert evaluate_multi_regime_hf(bars_up[:10]) == ("SEM SINAL", "mercado_sem_oscilacao")
    assert evaluate_multi_regime_hf(_compressed_bars(10, 45, bull_last=True)) == ("SEM SINAL", "virada_de_hora")
    monkeypatch.setattr("src.indicator.strategy_multi_regime._wilder_atr", lambda _b, _p: 0.0)
    assert evaluate_multi_regime_hf(bars_up) == ("SEM SINAL", "mercado_sem_oscilacao")
    monkeypatch.undo()

    neutral = [Candle(_epoch(10, 0) - (54 - i) * 900, 100.0, 101.0, 99.0, 100.0) for i in range(55)]
    assert evaluate_multi_regime_hf(neutral, "OTC_GDAXI", None) == ("SEM SINAL", "sem_setup_multi_regime")

    monkeypatch.setattr("src.indicator.strategy_multi_regime._eval_climax_global", lambda *a: "PUT")
    monkeypatch.setattr("src.indicator.strategy_multi_regime._eval_di5_pullback", lambda *a: "CALL")
    monkeypatch.setattr("src.indicator.strategy_multi_regime._eval_q2_cci14", lambda *a: None)
    assert evaluate_multi_regime_hf(bars_up, "OTC_GDAXI", peers) == ("SEM SINAL", "conflito_multi_regime")


@pytest.mark.asyncio
async def test_pullback_training_causal_peer_slice_and_multi_regime(monkeypatch):
    bars = _compressed_bars(10, 0, bull_last=True)
    extra = [Candle(bars[-1].epoch + (i + 1) * 900, 100.0, 101.0, 99.0, 100.2) for i in range(150)]
    gap_tail = [Candle(extra[-1].epoch + (i + 5) * 900, 100.0, 101.0, 99.0, 100.2) for i in range(10)]
    full = bars + extra + gap_tail
    cursors: dict[str, int] = {}
    sl = _causal_peer_slice({"OTC_GDAXI": full}, cursors, full[55].epoch)
    assert "OTC_GDAXI" in sl
    assert sl["OTC_GDAXI"][-1].epoch == full[55].epoch

    class _Source:
        async def symbols(self):
            return [
                {"market": "indices", "underlying_symbol": "OTC_GDAXI", "underlying_symbol_type": "stockindex"},
                {"market": "indices", "underlying_symbol": "OTC_FCHI", "underlying_symbol_type": "stockindex"},
            ]

    class _Store:
        async def set_symbols(self, symbols):
            self.symbols = symbols

        async def register_setups(self, symbols, version):
            self.version = version

        async def save_candles(self, symbol, period, candles):
            return None

    async def fake_history(_source, _symbol, _now, period):
        return full if period == 900 else []

    monkeypatch.setattr("src.indicator.pullback_training.history", fake_history)
    report = await diagnose_all(
        _Source(), _Store(), now=full[-1].epoch + 900, variant="multi_regime_hf", scope="stock_indices"
    )
    assert report["OTC_GDAXI:900"]["estado"] == "PROXY DE VELAS"
    assert report["OTC_GDAXI:900"]["versao"] == version_for_variant("multi_regime_hf")
