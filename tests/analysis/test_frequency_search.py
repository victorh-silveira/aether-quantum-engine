"""Pesquisa integrada com sinais sinteticos e gates reais de replay e inferencia."""

import hashlib
import json
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from scripts.analysis.frequency_replay import Signal
from scripts.analysis.frequency_search import cache_signals, run
from src.indicator.frequency_candidate import Candidate, generate_candidates


BASELINE = Candidate().label
FASTER = Candidate(branches=(1,), di_scale=0.5).label


def _epoch(day):
    """Converte as fronteiras fixas UTC da amostra sintetica de 2026."""
    return int(datetime.fromisoformat(day).replace(tzinfo=UTC).timestamp())


def _signals():
    """Constroi duas hipoteses; vantagem futura nao existe no treino inicial."""
    baseline, faster = [], []
    current = datetime(2026, 1, 1, tzinfo=UTC)
    cutoff = datetime(2026, 11, 25, tzinfo=UTC)
    while current <= cutoff:
        epoch = int(current.timestamp()) + 9 * 3600
        baseline.extend(Signal("OTC_FTSE", epoch + index * 900, "CALL", 1) for index in range(3))
        outcome = -1 if current.month < 4 else 1
        faster.extend(Signal("OTC_FTSE", epoch + index * 900, "CALL", outcome) for index in range(4))
        current += timedelta(days=1)
    return {BASELINE: baseline, FASTER: faster}


def _freeze(tmp_path, *, calendar_complete=False):
    """Grava dataset real minimo e manifesto verificavel, sem coletar rede."""
    path = tmp_path / "candles.parquet"
    pl.DataFrame(
        {
            "symbol": ["OTC_FTSE"],
            "period": [900],
            "epoch": [_epoch("2026-01-01")],
            "open": [100.0],
            "high": [102.0],
            "low": [99.0],
            "close": [101.0],
        }
    ).write_parquet(path)
    manifest = {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "start_epoch": _epoch("2026-01-01T03:00:00"),
        "cutoff_epoch": _epoch("2026-11-25T20:00:00"),
        "calendar": {"coverage_complete": calendar_complete},
        "qualifica_validacao_anual_completa": calendar_complete,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "settings_baseline.json").write_text(
        json.dumps({"economic_calendar": {"enabled": True}}), encoding="utf-8"
    )
    (tmp_path / "economic_calendar_snapshot.json").write_text(json.dumps({"events": []}), encoding="utf-8")
    return manifest


def _mock_generation(monkeypatch, *, full_family=False):
    """Substitui apenas geracao de sinais; selecao, estatistica e replay sao reais."""
    signals = _signals()
    if full_family:
        signals = {candidate.label: signals.get(candidate.label, []) for candidate in generate_candidates()}

    def persistent(dataset, start, cutoff):
        assert ("OTC_FTSE", 900) in dataset
        assert start == _epoch("2026-01-01T03:00:00")
        assert cutoff == _epoch("2026-11-25T20:00:00")
        return signals, {"synthetic": True}

    def comparators(dataset, start, cutoff):
        return (
            {label: [] for label in ("dual_sniper", "multi_regime_hf", "senior_institutional")} if full_family else {}
        )

    monkeypatch.setattr("scripts.analysis.frequency_search.persistent_signals", persistent)
    monkeypatch.setattr("scripts.analysis.frequency_search.comparator_signals", comparators)


def test_run_never_qualifies_without_full_family_and_has_no_future_fold_leak(tmp_path, monkeypatch):
    _freeze(tmp_path, calendar_complete=True)
    _mock_generation(monkeypatch)
    report = run(tmp_path, samples=200)
    assert report["bootstrap_samples"] == 200
    assert report["calendar_coverage_complete"] is True
    assert report["identity"]["complete_family"] is False
    assert report["data_qualified"] is False
    assert report["winner"] is None
    assert report["qualifies_contract"] is False
    assert report["demo"]["enabled"] is False
    assert len(report["oos"]) == 2
    assert {scenario["payout"] for scenario in report["scenarios"][BASELINE]} == {0.7, 0.7235, 0.8}
    folds = report["walk_forward"]
    assert [fold["month"] for fold in folds] == [f"2026-{month:02d}" for month in range(4, 11)]
    assert folds[0]["statistical_choice_only"]["label"] == BASELINE
    for month, fold in zip(range(4, 11), folds, strict=True):
        assert fold["training_end"] == _epoch(f"2026-{month:02d}-01T03:00:00") - 1
        assert all(day.startswith(f"2026-{month:02d}-") for day in fold["test"]["daily"])
        assert fold["test"]["qualifies_contract"] is False
    stored = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert stored["winner"] is None
    assert json.loads((tmp_path / "demo_candidate.json").read_text())["required_settled_contracts"] == 200
    assert len(json.loads((tmp_path / "candidate_grid.json").read_text())) == 145


@pytest.mark.parametrize("inconsistent_claim", [False, True])
def test_full_family_still_cannot_qualify_without_calendar(tmp_path, monkeypatch, inconsistent_claim):
    manifest = _freeze(tmp_path)
    manifest["qualifica_validacao_anual_completa"] = inconsistent_claim
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _mock_generation(monkeypatch, full_family=True)
    report = run(tmp_path, samples=200)
    assert report["identity"]["complete_family"] is True
    assert report["calendar_coverage_complete"] is False
    assert report["data_qualified"] is False
    assert len(report["oos"]) == 148
    assert report["winner"] is None
    assert report["demo"]["enabled"] is False


def test_dataset_checksum_corruption_fails_before_generation(tmp_path, monkeypatch):
    _freeze(tmp_path)
    (tmp_path / "candles.parquet").write_bytes(b"dataset adulterado")

    def forbidden(*args, **kwargs):
        pytest.fail("nao deve gerar sinais para dataset adulterado")

    monkeypatch.setattr("scripts.analysis.frequency_search.persistent_signals", forbidden)
    with pytest.raises(ValueError, match="Dataset alterado"):
        run(tmp_path, samples=200)


def test_cached_signal_checksum_corruption_fails_closed(tmp_path, monkeypatch):
    manifest = _freeze(tmp_path)
    _mock_generation(monkeypatch, full_family=True)
    signals, identity = cache_signals(tmp_path, manifest, stage="all")
    assert len(signals) == 148
    assert identity["complete_family"] is True
    cache = tmp_path / "signals" / f"{BASELINE}.parquet"
    cache.write_bytes(cache.read_bytes() + b"corrompido")
    with pytest.raises(ValueError, match="Hash de sinais divergente"):
        cache_signals(tmp_path, manifest, stage="all")


def test_staged_cache_remains_reusable_across_groups(tmp_path, monkeypatch):
    manifest = _freeze(tmp_path)
    _mock_generation(monkeypatch, full_family=True)
    persistent, _ = cache_signals(tmp_path, manifest, stage="persistent")
    comparators, _ = cache_signals(tmp_path, manifest, stage="comparators")
    signals, identity = cache_signals(tmp_path, manifest, stage="all")
    assert len(persistent) == 145
    assert len(comparators) == 3
    assert len(signals) == 148
    assert identity["complete_family"] is True


@pytest.mark.parametrize("alteration", ["corrupt", "missing"])
def test_partial_stage_cannot_reseal_corruption_in_other_group(tmp_path, monkeypatch, alteration):
    manifest = _freeze(tmp_path)
    _mock_generation(monkeypatch, full_family=True)
    cache_signals(tmp_path, manifest, stage="all")
    index = tmp_path / "signals" / "index.json"
    original_index = index.read_bytes()
    other_group = tmp_path / "signals" / "dual_sniper.parquet"
    if alteration == "corrupt":
        other_group.write_bytes(other_group.read_bytes() + b"adulterado")
    else:
        other_group.unlink()
    with pytest.raises(ValueError, match="Hash de sinais divergente: dual_sniper"):
        cache_signals(tmp_path, manifest, stage="persistent")
    assert index.read_bytes() == original_index


def test_partial_stage_preserves_original_hashes_of_other_group(tmp_path, monkeypatch):
    manifest = _freeze(tmp_path)
    _mock_generation(monkeypatch, full_family=True)
    cache_signals(tmp_path, manifest, stage="all")
    index = tmp_path / "signals" / "index.json"
    original = json.loads(index.read_text())
    cache_signals(tmp_path, manifest, stage="persistent")
    refreshed = json.loads(index.read_text())
    assert refreshed["hashes"] == original["hashes"]
    paths = {path.replace("\\", "/") for path in refreshed["code_hashes"]}
    assert "src/indicator/institutional.py" in paths
    assert "src/indicator/pullback_trend_regime.py" in paths
