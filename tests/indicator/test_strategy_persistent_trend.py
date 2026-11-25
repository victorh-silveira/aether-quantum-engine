"""Testes unitarios da estrategia persistent_trend_v3 (Wilder DI + H1 EMA20/50)."""

from __future__ import annotations

from src.indicator.domain import Candle
from src.indicator.pullback import evaluate, version_for_variant
from src.indicator.strategy_persistent_trend import (
    _branch_di_adx,
    _branch_di_continuation,
    _branch_di_pullback,
    _branch_h1_di,
    _candle_dir,
    _resample_h1_closes_causal,
    _resolve_h1_trend,
    calc_wilder_di_adx,
    evaluate_persistent_trend_v3,
)


def _trend_bars(
    count: int = 80,
    *,
    step: float = 2.0,
    start_epoch: int = 1_767_225_600,
    last_reverse: bool = False,
) -> list[Candle]:
    """Gera serie M15 sintetica direcional fora das horas toxicas UTC."""
    bars: list[Candle] = []
    price = 1000.0
    for idx in range(count):
        epoch = start_epoch + idx * 900
        delta = -step if (last_reverse and idx == count - 1) else step
        op = price
        cl = price + delta
        hi = max(op, cl) + 0.5
        lo = min(op, cl) - 0.5
        bars.append(Candle(epoch, op, hi, lo, cl))
        price = cl
    return bars


def _h1_bars(count: int = 60, *, step: float = 4.0, end_epoch: int = 1_767_300_000) -> list[Candle]:
    """Gera velas H1 fechadas com tendencia controlada."""
    start = ((end_epoch // 3600) - count) * 3600
    bars: list[Candle] = []
    price = 1000.0
    for idx in range(count):
        op = price
        cl = price + step
        bars.append(Candle(start + idx * 3600, op, max(op, cl) + 1.0, min(op, cl) - 1.0, cl))
        price = cl
    return bars


def test_version_for_persistent_trend_v3():
    assert version_for_variant("persistent_trend_v3").startswith("pullback-1-")


def test_calc_wilder_di_adx_edge_and_directional_cases():
    bars_up = _trend_bars(40, step=3.0)
    bars_dn = _trend_bars(40, step=-3.0)
    flat = [Candle(1_767_225_600 + i * 900, 100.0, 100.0, 100.0, 100.0) for i in range(30)]
    no_dm = [Candle(1_767_225_600 + i * 900, 100.0, 101.0, 99.0, 100.0) for i in range(30)]

    assert calc_wilder_di_adx(bars_up[:10], 14) == (0.0, 0.0)
    assert calc_wilder_di_adx(bars_up, 0) == (0.0, 0.0)
    assert calc_wilder_di_adx(flat, 14) == (0.0, 0.0)
    di_nodm, adx_nodm = calc_wilder_di_adx(no_dm, 14)
    assert di_nodm == 0.0 and adx_nodm == 0.0

    di_up, adx_up = calc_wilder_di_adx(bars_up, 14)
    di_dn, adx_dn = calc_wilder_di_adx(bars_dn, 14)
    assert di_up > 50.0 and adx_up > 50.0
    assert di_dn < -50.0 and adx_dn > 50.0


def test_h1_resample_and_trend_resolution():
    assert _resample_h1_closes_causal([], 3600) == []
    # 220 velas M15 = 55 horas completas
    m15_up = _trend_bars(220, step=1.5)
    m15_dn = _trend_bars(220, step=-1.5)
    assert _resolve_h1_trend(m15_up, None) == 1
    assert _resolve_h1_trend(m15_dn, None) == -1
    assert _resolve_h1_trend(m15_up[:60], None) == 0

    h1_up = _h1_bars(60, step=3.0, end_epoch=m15_up[59].epoch + 900)
    h1_dn = _h1_bars(60, step=-3.0, end_epoch=m15_up[59].epoch + 900)
    assert _resolve_h1_trend(m15_up[:60], h1_up) == 1
    assert _resolve_h1_trend(m15_up[:60], h1_dn) == -1


def test_branch_helpers_and_candle_dir():
    assert _candle_dir(Candle(0, 10.0, 11.0, 9.0, 10.5)) == 1
    assert _candle_dir(Candle(0, 10.0, 11.0, 9.0, 9.5)) == -1
    assert _candle_dir(Candle(0, 10.0, 11.0, 9.0, 10.0)) == 0

    assert _branch_h1_di(1, 45.0, 40.0) == "CALL"
    assert _branch_h1_di(-1, -45.0, 40.0) == "PUT"
    assert _branch_h1_di(1, -45.0, 40.0) is None

    assert _branch_di_pullback(50.0, -1, 45.0) == "CALL"
    assert _branch_di_pullback(-50.0, 1, 45.0) == "PUT"
    assert _branch_di_pullback(50.0, 1, 45.0) is None

    assert _branch_di_continuation(40.0, 1, 38.0) == "CALL"
    assert _branch_di_continuation(-40.0, -1, 38.0) == "PUT"
    assert _branch_di_continuation(40.0, -1, 38.0) is None

    assert _branch_di_adx(45.0, 40.0, 35.0, 40.0) == "CALL"
    assert _branch_di_adx(-45.0, 40.0, 35.0, 40.0) == "PUT"
    assert _branch_di_adx(45.0, 20.0, 35.0, 40.0) is None
    assert _branch_di_adx(10.0, 40.0, 35.0, 40.0) is None


def test_evaluate_persistent_trend_v3_signals_guards_and_wrapper(monkeypatch):
    # Hora 00:00 UTC + 79*15m = 19:45 UTC (hora toxica 19 UTC!)
    bars_toxic = _trend_bars(80, step=2.0, start_epoch=1_767_225_600)
    assert (bars_toxic[-1].epoch // 3600) % 24 == 19
    assert evaluate_persistent_trend_v3(bars_toxic) == ("SEM SINAL", "hora_toxica_utc")

    # Curto ou vela sem oscilacao
    assert evaluate_persistent_trend_v3(bars_toxic[:20]) == ("SEM SINAL", "mercado_sem_oscilacao")
    flat_last = bars_toxic[:-1] + [Candle(bars_toxic[-1].epoch, 100.0, 100.0, 100.0, 100.0)]
    assert evaluate_persistent_trend_v3(flat_last) == ("SEM SINAL", "mercado_sem_oscilacao")

    # Hora valida (08:00 UTC = start + 8h)
    start_valid = 1_767_225_600 + 8 * 3600
    bars_up = _trend_bars(60, step=3.0, start_epoch=start_valid)
    bars_dn = _trend_bars(60, step=-3.0, start_epoch=start_valid)
    h1_up = _h1_bars(60, step=3.0, end_epoch=bars_up[-1].epoch + 900)
    h1_dn = _h1_bars(60, step=-3.0, end_epoch=bars_dn[-1].epoch + 900)

    assert evaluate_persistent_trend_v3(bars_up, h1_bars=h1_up) == ("CALL", "setup_confirmado")
    assert evaluate_persistent_trend_v3(bars_dn, h1_bars=h1_dn) == ("PUT", "setup_confirmado")

    # Integracao via pullback.evaluate
    res = evaluate(
        bars_up,
        900,
        bars_up[-1].epoch + 900,
        variant="persistent_trend_v3",
        allow_session_gaps=True,
        symbol="OTC_FTSE",
        h1_bars=h1_up,
    )
    assert res.side == "CALL"
    assert res.reason == "setup_confirmado"
    assert res.version == version_for_variant("persistent_trend_v3")

    # Mercado neutro (sem setup)
    neutral = [
        Candle(start_valid + i * 900, 100.0, 101.0, 99.0, 100.0 + (0.2 if i % 2 == 0 else -0.2)) for i in range(60)
    ]
    assert evaluate_persistent_trend_v3(neutral) == ("SEM SINAL", "sem_setup_persistent_trend")

    # Conflito entre ramos
    monkeypatch.setattr("src.indicator.strategy_persistent_trend._branch_h1_di", lambda *_args: "CALL")
    monkeypatch.setattr("src.indicator.strategy_persistent_trend._branch_di_adx", lambda *_args: "PUT")
    assert evaluate_persistent_trend_v3(bars_up, h1_bars=h1_up) == ("SEM SINAL", "conflito_persistent_trend")
