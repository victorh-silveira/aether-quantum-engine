"""Testes unitarios completos para a variante bimodal dual_sniper (T18 + C08)."""

from __future__ import annotations

from datetime import datetime

from src.indicator.domain import Candle
from src.indicator.institutional import BRT_TIMEZONE
from src.indicator.pullback import evaluate, version_for_variant
from src.indicator.pullback_dual_sniper import (
    _closed_h1_trend,
    _evaluate_c08_compression,
    _evaluate_pb09_pb22,
    _evaluate_t18_trend,
    _peer_directions_at_epoch,
    _sma_tr,
    _wilder_atr_adx,
    evaluate_dual_sniper,
)


def _make_trend_m15(count: int = 260, *, bull: bool = True, start_epoch: int | None = None) -> list[Candle]:
    base_ep = (
        start_epoch if start_epoch is not None else int(datetime(2026, 11, 10, 6, 0, tzinfo=BRT_TIMEZONE).timestamp())
    )
    base_ep = (base_ep // 3600) * 3600
    delta = 1.0 if bull else -1.0
    bars: list[Candle] = []
    price = 1000.0
    for i in range(count):
        o = price
        c = o + delta * 0.8
        hi = max(o, c) + 0.3
        lo = min(o, c) - 0.3
        bars.append(Candle(base_ep + i * 900, o, hi, lo, c))
        price = c
    return bars


def test_version_and_helpers_edge_cases():
    assert version_for_variant("dual_sniper").startswith("pullback-1-")
    assert _sma_tr([], 14) == 0.0
    assert _sma_tr([Candle(0, 1.0, 2.0, 0.5, 1.5)], 14) == 0.0
    assert _closed_h1_trend([]) == 0
    assert _closed_h1_trend([Candle(i * 900, 100.0, 101.0, 99.0, 100.0) for i in range(20)]) == 0
    assert _wilder_atr_adx([Candle(i * 900, 100.0, 101.0, 99.0, 100.0) for i in range(10)], 14) == (0.0, 0.0)
    flat = [Candle(i * 900, 100.0, 100.0, 100.0, 100.0) for i in range(60)]
    assert _wilder_atr_adx(flat, 14) == (0.0, 0.0)
    assert _closed_h1_trend([Candle(i * 900, 100.0, 101.0, 99.0, 100.0) for i in range(240)]) == 0
    assert (
        _evaluate_pb09_pb22([Candle(0, 100, 101, 99, 100), Candle(1800, 100, 101, 99, 100)], 102, 101, 100, 1.0) is None
    )
    assert _peer_directions_at_epoch("UNKNOWN", 900, {}) == []
    assert _peer_directions_at_epoch("OTC_GDAXI", 900, {"OTC_GDAXI": []}) == []
    assert _peer_directions_at_epoch("OTC_GDAXI", 900, {"OTC_GDAXI": [Candle(1800, 1, 2, 0.5, 1.5)]}) == []
    assert _peer_directions_at_epoch("OTC_GDAXI", 900, {"OTC_GDAXI": [Candle(900, 1, 2, 0.5, 1.0)]}) == [0]


def test_t18_trend_call_and_put_pb09_and_pb22():
    bull = _make_trend_m15(260, bull=True)
    # Impulso forte em -2 e micro-recuo acima da EMA9 em -1 (PB09)
    p2 = bull[-2]
    bull[-2] = Candle(p2.epoch, p2.open, p2.open + 4.5, p2.open - 0.1, p2.open + 4.2)
    p1 = bull[-1]
    bull[-1] = Candle(p1.epoch, bull[-2].close, bull[-2].close + 0.1, bull[-2].close - 1.2, bull[-2].close - 0.8)
    res_call = evaluate(
        bull, 900, bull[-1].epoch + 900, variant="dual_sniper", allow_session_gaps=True, symbol="OTC_SPC"
    )
    assert res_call.side == "CALL"
    assert res_call.reason == "setup_confirmado"

    # PB22 com pavio superior >= 25% e body_prev >= 1.0 ATR
    bull_pb22 = list(bull)
    bull_pb22[-2] = Candle(p2.epoch, p2.open, p2.open + 1.8, p2.open - 0.1, p2.open + 1.55)
    bull_pb22[-1] = Candle(
        p1.epoch, bull_pb22[-2].close, bull_pb22[-2].close + 1.2, bull_pb22[-2].close - 0.9, bull_pb22[-2].close - 0.72
    )
    assert (
        _evaluate_pb09_pb22(
            bull_pb22, bull_pb22[-1].close - 0.5, bull_pb22[-1].close - 1.5, bull_pb22[-1].close - 2.5, 1.4
        )
        == "CALL"
    )

    # Espelho PUT (PB09 + H1 baixa + ADX >= 25)
    bear = _make_trend_m15(260, bull=False)
    b2 = bear[-2]
    bear[-2] = Candle(b2.epoch, b2.open, b2.open + 0.1, b2.open - 4.5, b2.open - 4.2)
    b1 = bear[-1]
    bear[-1] = Candle(b1.epoch, bear[-2].close, bear[-2].close + 1.2, bear[-2].close - 0.1, bear[-2].close + 0.8)
    res_put = evaluate(
        bear, 900, bear[-1].epoch + 900, variant="dual_sniper", allow_session_gaps=True, symbol="OTC_NDX"
    )
    assert res_put.side == "PUT"
    assert res_put.reason == "setup_confirmado"

    # PB22 PUT com pavio inferior >= 25%
    bear_pb22 = list(bear)
    bear_pb22[-2] = Candle(b2.epoch, b2.open, b2.open + 0.1, b2.open - 1.8, b2.open - 1.55)
    bear_pb22[-1] = Candle(
        b1.epoch, bear_pb22[-2].close, bear_pb22[-2].close + 0.9, bear_pb22[-2].close - 1.2, bear_pb22[-2].close + 0.72
    )
    assert (
        _evaluate_pb09_pb22(
            bear_pb22, bear_pb22[-1].close + 0.5, bear_pb22[-1].close + 1.5, bear_pb22[-1].close + 2.5, 1.4
        )
        == "PUT"
    )

    # T18 retorna None quando H1 diverge ou ADX < 25 ou sem padrao
    assert (
        _evaluate_t18_trend(
            bull[:-1] + [Candle(p1.epoch, p1.open, p1.open + 5.0, p1.open - 5.0, p1.open + 4.0)],
            [b.close for b in bull[:-1]] + [p1.open + 4.0],
        )
        is None
    )
    weak = [Candle(i * 900, 100.0 + (i % 2) * 0.2, 100.5, 99.5, 100.0 + ((i + 1) % 2) * 0.2) for i in range(60)]
    assert _evaluate_t18_trend(weak, [b.close for b in weak]) is None


def test_c08_compression_put_call_and_guards(monkeypatch):
    # Constroi serie com ATR50 alto (primeiras 40 barras range=10) e ATR14 baixo (ultimas 13 barras range=2)
    base_ep = int(datetime(2026, 11, 10, 8, 0, tzinfo=BRT_TIMEZONE).timestamp())
    bars: list[Candle] = []
    for i in range(55):
        ep = base_ep + i * 900
        if i < 40:
            bars.append(Candle(ep, 100.0, 105.0, 95.0, 101.0))
        elif i < 54:
            bars.append(Candle(ep, 100.0, 101.0, 99.0, 100.2))
        else:
            # Barra forte esticada de alta (> 1.2 ATR14, body/rng > 0.55) -> fade PUT
            bars.append(Candle(ep, 100.0, 104.2, 99.8, 104.0))

    last_ep = bars[-1].epoch
    peers_up = {
        "OTC_GDAXI": [Candle(last_ep, 100, 104, 99, 103)],
        "OTC_FCHI": [Candle(last_ep, 100, 104, 99, 103)],
        "OTC_FTSE": [Candle(last_ep, 100, 104, 99, 103)],
        "OTC_SX5E": [Candle(last_ep, 100, 104, 99, 103)],
    }
    res_put = evaluate(
        bars, 900, last_ep + 900, variant="dual_sniper", allow_session_gaps=True, symbol="OTC_GDAXI", peer_bars=peers_up
    )
    assert res_put.side == "PUT"
    assert res_put.reason == "setup_confirmado"

    # Espelho CALL (barra forte de baixa + bloco em baixa)
    bars_dn = list(bars)
    bars_dn[-1] = Candle(last_ep, 104.0, 104.2, 99.8, 100.0)
    peers_dn = {k: [Candle(last_ep, 104, 105, 100, 101)] for k in peers_up}
    assert _evaluate_c08_compression(bars_dn, "OTC_GDAXI", peers_dn) == "CALL"

    # Guardas do C08: simbolo fora dos blocos, vol_ratio >= 0.90, body_ratio <= 0.55, ret_atr <= 1.2, <3 pares, consenso < 75%
    assert _evaluate_c08_compression(bars, "R_100", peers_up) is None
    us_peers = {
        "OTC_SPC": [Candle(last_ep, 100, 104, 99, 103)],
        "OTC_NDX": [Candle(last_ep, 100, 104, 99, 103)],
        "OTC_DJI": [Candle(last_ep, 100, 104, 99, 103)],
    }
    assert _evaluate_c08_compression(bars, "OTC_SPC", us_peers) == "PUT"
    ny_ep = int(datetime(2026, 11, 10, 11, 15, tzinfo=BRT_TIMEZONE).timestamp())
    bars_ny = [Candle(ny_ep - (54 - i) * 900, b.open, b.high, b.low, b.close) for i, b in enumerate(bars)]
    peers_ny = {k: [Candle(ny_ep, 100, 104, 99, 103)] for k in peers_up}
    assert _evaluate_c08_compression(bars_ny, "OTC_GDAXI", peers_ny) == "PUT"
    assert (
        _evaluate_c08_compression([Candle(i * 900, 100, 100, 100, 100) for i in range(55)], "OTC_GDAXI", peers_up)
        is None
    )
    high_vol = [Candle(base_ep + i * 900, 100.0, 105.0, 95.0, 104.0) for i in range(55)]
    assert _evaluate_c08_compression(high_vol, "OTC_GDAXI", peers_up) is None
    small_body = list(bars)
    small_body[-1] = Candle(last_ep, 100.0, 108.0, 92.0, 104.0)
    assert _evaluate_c08_compression(small_body, "OTC_GDAXI", peers_up) is None
    small_ret = list(bars)
    small_ret[-1] = Candle(last_ep, 100.0, 101.0, 99.9, 100.9)
    assert _evaluate_c08_compression(small_ret, "OTC_GDAXI", peers_up) is None
    assert _evaluate_c08_compression(bars, "OTC_GDAXI", {"OTC_GDAXI": peers_up["OTC_GDAXI"]}) is None
    split_peers = {
        "OTC_GDAXI": [Candle(last_ep, 100, 104, 99, 103)],
        "OTC_FCHI": [Candle(last_ep, 104, 105, 100, 101)],
        "OTC_FTSE": [Candle(last_ep, 104, 105, 100, 101)],
        "OTC_SX5E": [Candle(last_ep, 100, 104, 99, 103)],
    }
    assert _evaluate_c08_compression(bars, "OTC_GDAXI", split_peers) is None

    # Conflito bimodal, sem setup e mercado sem oscilacao
    assert evaluate_dual_sniper(bars[:10], [b.close for b in bars[:10]]) == ("SEM SINAL", "mercado_sem_oscilacao")
    assert evaluate_dual_sniper(bars, [b.close for b in bars], "OTC_SPC", None) == ("SEM SINAL", "sem_setup_sniper")
    monkeypatch.setattr("src.indicator.pullback_dual_sniper._evaluate_t18_trend", lambda b, c, h1_bars=None: "CALL")
    monkeypatch.setattr("src.indicator.pullback_dual_sniper._evaluate_c08_compression", lambda b, s, p: "PUT")
    assert evaluate_dual_sniper(bars, [b.close for b in bars], "OTC_GDAXI", peers_up) == (
        "SEM SINAL",
        "conflito_bimodal",
    )

    # T18 com H1 nao alinhado retorna None
    monkeypatch.setattr("src.indicator.pullback_dual_sniper._closed_h1_trend", lambda b, h1_bars=None: -1)
    monkeypatch.setattr("src.indicator.pullback_dual_sniper._wilder_atr_adx", lambda b, p=14: (1.0, 30.0))
    monkeypatch.setattr("src.indicator.pullback_dual_sniper._evaluate_pb09_pb22", lambda *a: "CALL")
    assert _evaluate_t18_trend(bars, [b.close for b in bars]) is None


def test_direct_h1_bars_unlock_t18_and_service_h1_cache():
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    from src.indicator.pullback_service import PullbackService

    # Apenas 80 velas M15 (< 51 H1 agregadas), mas 65 barras H1 diretas fechadas
    short_m15 = _make_trend_m15(80, bull=True)
    assert _closed_h1_trend(short_m15) == 0

    end_ts = short_m15[-1].epoch + 900
    h1_start = ((end_ts // 3600) - 65) * 3600
    h1_bull = [Candle(h1_start + i * 3600, 1000.0 + i, 1002.0 + i, 999.0 + i, 1001.0 + i) for i in range(66)]
    assert _closed_h1_trend(short_m15, h1_bars=h1_bull) == 1

    # Valida PullbackService aquecendo _h1_cache automaticamente em dual_sniper
    class _FakeSource:
        async def symbols(self):
            return [{"symbol": "OTC_SSMI", "display_name": "Swiss 20", "market": "indices", "exchange_is_open": 1}]

        async def candles(self, symbol, count=500, period=900):
            base = (1700000000 // period) * period - 60 * period
            return [
                {"epoch": base + i * period, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5}
                for i in range(60)
            ]

    sink = MagicMock()
    sink.set_symbols = AsyncMock()
    sink.register_setups = AsyncMock()
    sink.promote_setups = AsyncMock()
    sink.save_setup = AsyncMock()
    metrics = MagicMock()
    svc = PullbackService(_FakeSource(), sink, metrics, variant="dual_sniper", scope="stock_indices")
    asyncio.run(svc.tick(1700000000 + 3600))
    assert "OTC_SSMI" in svc._h1_cache
    assert len(svc._h1_cache["OTC_SSMI"]) > 0
