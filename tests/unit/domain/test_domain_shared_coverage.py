"""Testes complementares para 100% de cobertura em domain e shared."""

from __future__ import annotations

import importlib
import json
import logging
import sys
from datetime import UTC, datetime
from unittest.mock import mock_open, patch

import pytest

from src.domain import config_knobs
from src.domain.indicators.technical_indicators import compute_indicators
from src.domain.models.market_data import Candle
from src.domain.risk.risk_manager import RiskManager
from src.shared import paths, settings_io


def test_config_knobs_validation_and_merge():
    with (
        patch("src.domain.config_knobs.settings_io.load_settings_json", return_value=["bad"]),
        pytest.raises(ValueError, match="settings.json invalido"),
    ):
        config_knobs.load_settings_json()
    with patch("src.domain.config_knobs.settings_io.load_settings_json", return_value={"ok": 1}):
        assert config_knobs.load_settings_json() == {"ok": 1}

    with pytest.raises(ValueError, match="root.sub obrigatorio"):
        config_knobs.require_mapping(None, "sub", ("a",), "root")
    with pytest.raises(ValueError, match="root.sub incompleto"):
        config_knobs.require_mapping({"sub": {"a": 1}}, "sub", ("a", "b"), "root")
    assert config_knobs.require_mapping({"sub": {"a": 1}}, "sub", ("a",), "root") == {"a": 1}

    with pytest.raises(ValueError, match="blk obrigatorio"):
        config_knobs.require_keys(None, ("x",), "blk")
    with pytest.raises(ValueError, match="blk incompleto"):
        config_knobs.require_keys({"x": 1}, ("x", "y"), "blk")
    assert config_knobs.require_bool({"flag": 1}, "flag") is True

    merged = config_knobs.deep_merge({"a": {"b": 1, "c": 2}, "d": 3}, {"a": {"c": 9}, "e": 5})
    assert merged == {"a": {"b": 1, "c": 9}, "d": 3, "e": 5}

    with patch(
        "src.domain.config_knobs.load_settings_json", return_value={"laya": {"enabled": True, "nested": {"x": 1}}}
    ):
        with pytest.raises(ValueError, match="laya.missing obrigatorio"):
            config_knobs.merge_settings_block(("laya", "missing"), None)
        assert config_knobs.merge_settings_block(("laya",), None) == {"enabled": True, "nested": {"x": 1}}
        assert config_knobs.merge_settings_block(("laya",), {"nested": {"y": 2}}) == {
            "enabled": True,
            "nested": {"x": 1, "y": 2},
        }


def test_risk_manager_open_settle_and_restore_branches():
    rm = RiskManager({"params": {"stake_min": 1.0}}, logger=logging.getLogger("AETH"))
    rm.set_initial_bankroll(500.0)
    assert rm.initial_bankroll == 500.0
    rm.register_open_contract(101, "R_10", 10.0)
    rm.register_open_contract(101, "R_10", 10.0)
    assert rm.active_contract_ids == [101]

    pnl = rm.on_trade_settled(101, 9.5, symbol="R_10")
    assert pnl == 9.5
    assert rm.active_contract_ids == []
    assert rm.total_session_profit == 9.5
    assert rm.peak_session_profit == 9.5

    exported = rm.get_state()
    assert exported == {"initial_bankroll": 500.0, "total_session_profit": 9.5, "peak_session_profit": 9.5}
    rm.reset_session(600.0)
    assert rm.initial_bankroll == 600.0
    assert rm.total_session_profit == 0.0

    # Contrato nao registrado e payload valido/invalido no restore
    assert rm.on_trade_settled(999, -5.0) == -5.0
    rm.restore_state(None)
    assert rm.total_session_profit == -5.0
    rm.restore_state({"initial_bankroll": 750.0, "total_session_profit": 12.0, "peak_session_profit": 15.0})
    assert rm.initial_bankroll == 750.0
    assert rm.total_session_profit == 12.0
    assert rm.peak_session_profit == 15.0


def test_technical_indicators_flat_series_covers_zero_tr_and_std():
    bars = [
        Candle(
            "R_10",
            100.0,
            100.0,
            100.0,
            100.0,
            datetime.fromtimestamp(1_764_000_000 + i * 300, tz=UTC),
            1_764_000_000 + i * 300,
        )
        for i in range(25)
    ]
    snap = compute_indicators(bars)
    assert snap.adx_14 == 0.0
    assert snap.bollinger_pct_b == 0.5
    assert snap.bollinger_width == 0.0
    assert compute_indicators([bars[0]]).session_zscore == 0.0


def test_shared_paths_and_settings_io_branches(monkeypatch):
    monkeypatch.setattr(sys, "path", [p for p in sys.path if str(paths.SRC_ROOT) != p])
    importlib.reload(paths)
    assert str(paths.SRC_ROOT) in sys.path

    with (
        patch("pathlib.Path.open", mock_open(read_data=json.dumps(["not", "a", "dict"]))),
        pytest.raises(ValueError, match="settings.json invalido"),
    ):
        settings_io.load_settings_json()
    with patch("pathlib.Path.open", mock_open(read_data=json.dumps({"anchor": "R_10"}))):
        assert settings_io.load_settings_json() == {"anchor": "R_10"}
