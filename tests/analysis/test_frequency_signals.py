"""Paridade dos sinais vetorizados com gatilhos e janelas de producao."""

from datetime import UTC, datetime

import numpy as np
import polars as pl

from scripts.analysis.frequency_signals import _usable, candidate_sides, persistent_signals, read_dataset
from src.indicator.domain import Candle
from src.indicator.frequency_candidate import Candidate, Features, evaluate_features, generate_candidates
from src.indicator.pullback import evaluate


EPOCH = int(datetime(2026, 4, 1, tzinfo=UTC).timestamp())


def _trajectory(n: int = 540) -> list[Candle]:
    """Cria alternancia deterministica de pressao forte com OHLC validos."""
    bars = []
    price = 1000.0
    for index in range(n):
        close = price + (1 if (index // 85) % 2 == 0 else -1)
        bars.append(Candle(EPOCH + index * 900, price, max(price, close) + 0.01, min(price, close) - 0.01, close))
        price = close
    return bars


def _h1() -> list[Candle]:
    """Inclui aquecimento H1 e velas futuras para testar corte causal."""
    return [Candle(EPOCH - 100 * 3600 + i * 3600, 100 + i, 102 + i, 99 + i, 101 + i) for i in range(300)]


def test_every_candidate_vector_matches_scalar_random_and_boundary_features():
    """Todas as 145 hipoteses respeitam fronteiras estritas e conflitos."""
    rng = np.random.default_rng(913)
    size = 250
    values = {name: rng.uniform(-70, 70, size) for name in ("di14", "di20", "di24", "di28")}
    values["adx20"] = rng.uniform(0, 70, size)
    values["h1_trend"] = rng.integers(-1, 2, size)
    values["direction"] = rng.integers(-1, 2, size)
    candidates = generate_candidates()
    assert len(candidates) == 145
    for candidate in candidates:
        boundaries = {
            "di14": [45 * candidate.di_scale, -45 * candidate.di_scale],
            "di20": [40 * candidate.di_scale, -40 * candidate.di_scale],
            "di24": [40 * candidate.di_scale, -40 * candidate.di_scale],
            "di28": [38 * candidate.di_scale, -38 * candidate.di_scale],
            "adx20": [candidate.adx_min, candidate.adx_min],
            "h1_trend": [1, -1],
            "direction": [1, -1],
        }
        features = {name: np.concatenate((array, boundaries[name])) for name, array in values.items()}
        actual = candidate_sides(features, candidate)
        expected = []
        for index in range(size + 2):
            side, _ = evaluate_features(Features(**{name: array[index] for name, array in features.items()}), candidate)
            expected.append({"CALL": 1, "PUT": -1, "SEM SINAL": 0}[side])
        np.testing.assert_array_equal(actual, expected, err_msg=candidate.label)


def test_usable_blocks_warmup_gaps_missing_outcomes_and_cutoff():
    """Somente entradas com duas velas previas e desfecho contiguos sao avaliadas."""
    bars = _trajectory(110)
    bars = [
        Candle(bar.epoch + (900 if i >= 75 else 0), bar.open, bar.high, bar.low, bar.close)
        for i, bar in enumerate(bars)
    ]
    eligible, audit = _usable(bars, bars[-1].epoch + 900, EPOCH)
    assert not eligible[:50].any()
    assert eligible[50:74].all()
    assert not eligible[74]
    assert not eligible[75]
    assert eligible[76:-1].all()
    assert not eligible[-1]
    assert audit["contiguous_evaluations"] == 57
    start_mask, _ = _usable(bars, bars[-1].epoch + 900, bars[90].epoch + 900)
    assert not start_mask[:90].any()
    assert start_mask[90]


def test_persistent_baseline_matches_full_production_evaluator_with_causal_h1():
    """V3 vetorizada conserva lados e outcomes apos 500 velas e horas H1 futuras."""
    bars, h1 = _trajectory(), _h1()
    cutoff = bars[-1].epoch + 900
    dataset = {("OTC_FTSE", 900): bars, ("OTC_FTSE", 3600): h1}
    output, audit = persistent_signals(dataset, EPOCH, cutoff)
    eligible, _ = _usable(bars, cutoff, EPOCH)
    expected = []
    for index in np.flatnonzero(eligible):
        closed_h1 = [bar for bar in h1 if bar.epoch + 3600 <= bars[index].epoch + 900][-500:]
        result = evaluate(
            bars[max(0, index - 499) : index + 1],
            900,
            bars[index].epoch + 900,
            "persistent_trend_v3",
            h1_bars=closed_h1,
        )
        if result.side in ("CALL", "PUT"):
            nxt = bars[index + 1]
            outcome = int(np.sign(nxt.close - nxt.open)) * (1 if result.side == "CALL" else -1)
            expected.append((bars[index].epoch + 900, result.side, outcome))
    baseline = output[Candidate().label]
    assert baseline
    assert [(signal.entry_epoch, signal.side, signal.outcome) for signal in baseline] == expected
    assert audit["OTC_FTSE"]["baseline_reasons"]["setup_confirmado"] == len(expected)
    assert len(output) == 145


def test_empty_and_h1_only_datasets():
    """Ausencia de M15 produz inventario vazio sem inventar sinais."""
    for dataset in ({}, {("A", 900): []}, {("A", 3600): _h1()}):
        output, audit = persistent_signals(dataset, EPOCH, EPOCH + 3600)
        assert not any(output.values())
        assert audit == {}
    eligible, inventory = _usable([], EPOCH + 3600, EPOCH)
    assert not eligible.any()
    assert inventory["candles"] == 0


def test_read_dataset_preserves_ohlc_order_and_partitions(tmp_path):
    """Parquet desordenado retorna series canonicas por ativo e periodo."""
    path = tmp_path / "candles.parquet"
    pl.DataFrame(
        [
            {"symbol": symbol, "period": period, "epoch": epoch, "open": 10.0, "high": 12.0, "low": 9.0, "close": 11.0}
            for symbol, period, epoch in (("A", 900, EPOCH + 900), ("B", 3600, EPOCH), ("A", 900, EPOCH))
        ]
    ).write_parquet(path)
    result = read_dataset(path)
    assert list(result) == [("A", 900), ("B", 3600)]
    assert [bar.epoch for bar in result[("A", 900)]] == [EPOCH, EPOCH + 900]
    assert result[("B", 3600)][0] == Candle(EPOCH, 10.0, 12.0, 9.0, 11.0)


def test_session_gap_reconciliation():
    """O sinal legado apos lacuna e identificado sem misturar a amostra contigua."""
    from scripts.analysis.frequency_search import _epoch
    from scripts.analysis.frequency_signals import session_gap_signals
    from src.indicator.domain import Candle

    start = _epoch("2026-11-09")
    bars = []
    for i in range(210):
        epoch = start + i * 900 + (37800 if i >= 180 else 0)
        value = 100 + i
        bars.append(Candle(epoch, value, value + 1.2, value - 0.2, value + 1))
    rows = session_gap_signals({("OTC_GDAXI", 900): bars}, start, _epoch("2026-12-01"))
    assert len(rows) == 1
    assert rows[0]["previous_gap_seconds"] == 38700
    assert rows[0]["outcome"] == 1
    assert rows[0]["side"] == "CALL"
