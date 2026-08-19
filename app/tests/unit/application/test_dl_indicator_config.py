from io import StringIO

import pytest

import src.application.services.deep_learning.dl_indicator_config as indicator_config_mod
from src.application.services.deep_learning.dl_feature_indicators import feature_windows
from src.application.services.deep_learning.dl_indicator_config import (
    indicator_windows,
    indicators_from_config,
    load_bb_width_anomaly_ratio,
    load_indicator_config_from_settings,
    reset_indicator_config_cache,
    resolve_indicator_config,
)
from src.application.services.deep_learning.dl_params import resolve_inference_history_bars


def test_resolve_indicator_config_from_settings():
    reset_indicator_config_cache()
    cfg = load_indicator_config_from_settings()
    assert cfg["windows"]["rsi_period"] == 7
    assert cfg["multipliers"]["bb_std_mult"] == 2.0
    assert cfg["congestion"]["min_bars"] == 67
    assert set(indicator_windows(cfg)) == set(cfg["windows"])


def test_resolve_indicator_config_rejects_incomplete():
    with pytest.raises(ValueError, match="obrigatorio|incompleto"):
        resolve_indicator_config({"indicators": {"windows": {"rsi_period": 14}}})
    with pytest.raises(ValueError, match="obrigatorio"):
        resolve_indicator_config({})
    with pytest.raises(ValueError, match="obrigatorio"):
        resolve_indicator_config({"indicators": {"windows": "bad"}})
    with pytest.raises(ValueError, match="obrigatorio"):
        indicator_windows({})


def test_load_bb_width_anomaly_ratio_missing(monkeypatch):
    class _FakePath:
        def open(self, *_args, **_kwargs):
            return StringIO('{"orchestrator": {"execution": {"bb_width_adaptive_squeeze": {}}}}')

    monkeypatch.setattr(indicator_config_mod, "repo_path", lambda *a, **k: _FakePath())
    with pytest.raises(KeyError, match="anomaly_ratio"):
        load_bb_width_anomaly_ratio()


def test_load_bb_width_anomaly_ratio():
    assert load_bb_width_anomaly_ratio() > 0.0


def test_resolve_inference_history_bars_uses_indicator_windows():
    base = load_indicator_config_from_settings()
    n = resolve_inference_history_bars(
        {"lookback": 30, "implied_vol_bars": 60, "indicators": {"windows": base["windows"]}},
        granularity=60,
    )
    assert n > 30
    lookback = 720
    n_m15 = resolve_inference_history_bars(
        {
            "lookback": lookback,
            "implied_vol_bars": 120,
            "indicators": {"windows": base["windows"]},
        },
        granularity=900,
    )
    assert n_m15 < 2 * lookback
    assert n_m15 >= lookback + 16
    assert n_m15 <= lookback + 128 + 16


def test_indicators_from_config_uses_embedded_block():
    reset_indicator_config_cache()
    base = load_indicator_config_from_settings()
    cfg = indicators_from_config(
        {
            "deep_learning": {
                "indicators": {
                    "windows": base["windows"],
                    "multipliers": base["multipliers"],
                    "normalization": base["normalization"],
                    "trend_consensus": base["trend_consensus"],
                    "congestion": base["congestion"],
                    "market_rank": base["market_rank"],
                    "vol_burst": base["vol_burst"],
                    "edge_zscore": base["edge_zscore"],
                    "exhaustion_filter": base["exhaustion_filter"],
                }
            }
        }
    )
    assert cfg["windows"]["rsi_period"] == base["windows"]["rsi_period"]
    assert indicators_from_config(None)["windows"]["bb_window"] == base["windows"]["bb_window"]


def test_feature_windows_reads_settings():
    reset_indicator_config_cache()
    win = feature_windows(60)
    assert win["bb_window"] == load_indicator_config_from_settings()["windows"]["bb_window"]
    assert win["rsi_period"] == load_indicator_config_from_settings()["windows"]["rsi_period"]
