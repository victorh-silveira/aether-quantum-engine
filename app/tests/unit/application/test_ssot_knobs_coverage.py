import json
from io import StringIO
from unittest.mock import patch

import pytest

import src.application.services.deep_learning.dl_calibration as dl_calib_mod
import src.application.services.deep_learning.dl_calibration_tolerance as dl_tol_mod
import src.application.services.deep_learning.dl_training_epochs as dl_epochs_mod
import src.application.services.execution_quality_gate_config as qg_cfg_mod
import src.application.services.execution_runtime_config as exec_rt_mod
import src.application.services.infra_timing_config as infra_mod
import src.application.services.live_signal_metrics_config as live_mod
import src.domain.config_knobs as knobs
from src.application.services.deep_learning.dl_params_blocks import parse_calibration_config
from src.application.services.execution_price_zone_gate import (
    apply_price_zone_gate,
    resolve_price_zone_config,
)
from src.application.services.execution_quality_gate_config import (
    quality_gate_from_config,
    resolve_quality_gate_config,
)
from src.application.services.execution_runtime_config import (
    reset_execution_runtime_cache,
    resolve_loss_protection_config,
    resolve_side_equilibrium_config,
)
from src.application.services.infra_timing_config import (
    resolve_history_fetch_config,
    resolve_stream_reconnect_config,
    resolve_triton_infer_timeout,
)
from src.application.services.live_signal_metrics_config import (
    load_live_signal_metrics_from_settings,
    reset_live_signal_metrics_config_cache,
    resolve_live_signal_metrics_config,
)
from src.domain.models.trade import TradeDirection


class _FakePath:
    def __init__(self, payload):
        self._payload = payload

    def open(self, *_a, **_k):
        text = self._payload if isinstance(self._payload, str) else json.dumps(self._payload)
        return StringIO(text)


def test_config_knobs_error_and_merge_paths(monkeypatch):
    with pytest.raises(ValueError, match="obrigatorio"):
        knobs.require_mapping({"x": 1}, "x", ("a",), "p")
    with pytest.raises(ValueError, match="incompleto"):
        knobs.require_mapping({"x": {}}, "x", ("a",), "p")
    with pytest.raises(ValueError, match="obrigatorio"):
        knobs.require_keys(None, ("a",), "p")
    with pytest.raises(ValueError, match="incompleto"):
        knobs.require_keys({"a": 1}, ("a", "b"), "p")
    assert knobs.deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 9}})["a"] == {"b": 1, "c": 9}
    monkeypatch.setattr(knobs, "repo_path", lambda *a, **k: _FakePath("[1,2,3]"))
    with pytest.raises(ValueError, match="invalido"):
        knobs.load_settings_json()
    monkeypatch.setattr(knobs, "repo_path", lambda *a, **k: _FakePath({"orchestrator": 1}))
    with pytest.raises(ValueError, match="obrigatorio"):
        knobs.merge_settings_block(("orchestrator", "execution"), None)


def test_infra_timing_error_and_override_paths(monkeypatch):
    monkeypatch.setattr(infra_mod, "repo_path", lambda *a, **k: _FakePath("[1]"))
    with pytest.raises(ValueError, match="invalido"):
        infra_mod._load_settings()
    cfg = resolve_stream_reconnect_config({"stream_reconnect": {"max_attempts": 9}})
    assert cfg["max_attempts"] == 9
    cfg2 = resolve_stream_reconnect_config(
        {"max_attempts": 7, "initial_backoff_seconds": 1.0, "max_backoff_seconds": 2.0}
    )
    assert cfg2["max_attempts"] == 7
    assert resolve_history_fetch_config({"history_fetch": {"chunk": 50}})["chunk"] == 50
    assert resolve_history_fetch_config({"chunk": 40})["chunk"] == 40
    with pytest.raises(ValueError, match="infer_timeout"):
        resolve_triton_infer_timeout({"triton": {}})


def test_quality_gate_config_branches(monkeypatch):
    monkeypatch.setattr(qg_cfg_mod, "repo_path", lambda *a, **k: _FakePath({"orchestrator": {"execution": {}}}))
    with pytest.raises(ValueError, match="quality_gate"):
        qg_cfg_mod._settings_quality_gate()
    monkeypatch.undo()
    full = resolve_quality_gate_config(None)
    assert resolve_quality_gate_config(full)["min_adx_threshold"] == full["min_adx_threshold"]
    assert resolve_quality_gate_config({})["min_payoff_edge"] == full["min_payoff_edge"]
    assert quality_gate_from_config(None)["min_adx_threshold"] == full["min_adx_threshold"]
    nested = quality_gate_from_config({"execution": {"quality_gate": {"min_adx_threshold": 1.0}}})
    assert nested["min_adx_threshold"] == 1.0
    orch = quality_gate_from_config({"orchestrator": {"execution": {"quality_gate": {"min_adx_threshold": 2.0}}}})
    assert orch["min_adx_threshold"] == 2.0


def test_execution_runtime_error_reset_and_side_eq(monkeypatch):
    reset_execution_runtime_cache()
    monkeypatch.setattr(exec_rt_mod, "repo_path", lambda *a, **k: _FakePath({"orchestrator": {}}))
    with pytest.raises(ValueError, match="execution obrigatorio"):
        exec_rt_mod._load_execution_from_settings()
    reset_execution_runtime_cache()
    assert exec_rt_mod._CACHE["execution"] is None
    monkeypatch.undo()
    reset_execution_runtime_cache()
    with pytest.raises(ValueError, match="disconnect obrigatorio"):
        resolve_loss_protection_config(
            {
                "loss_protection": {
                    "min_direction_margin": 0.1,
                    "recovery_min_direction_margin": 0.1,
                    "recovery_min_hurst": 0.1,
                    "max_edge_without_margin": 0.1,
                    "max_zscore_without_margin": 0.1,
                    "disconnect": "bad",
                }
            }
        )
    crafted = dict(exec_rt_mod._load_execution_from_settings())
    lp = dict(crafted["loss_protection"])
    disc = dict(lp["disconnect"])
    disc.pop("block_threshold", None)
    lp["disconnect"] = disc
    crafted["loss_protection"] = lp
    with (
        patch.object(exec_rt_mod, "_execution_block", return_value=crafted),
        pytest.raises(ValueError, match="block_threshold"),
    ):
        resolve_loss_protection_config(None)
    side = resolve_side_equilibrium_config(None)
    assert side["enabled"] is True
    assert "break_even_wr" in side


def test_price_zone_invalid_weights_and_indicator_telemetry():
    with (
        patch(
            "src.application.services.execution_price_zone_gate.merge_settings_block",
            return_value={"bb_weight": 0.0, "keltner_weight": 0.0},
        ),
        pytest.raises(ValueError, match="invalidos"),
    ):
        resolve_price_zone_config(
            {
                "price_zone": {
                    "enabled": True,
                    "buy_max": 0.3,
                    "sell_min": 0.7,
                    "bb_weight": 0.0,
                    "keltner_weight": 0.0,
                    "require_trend_agreement": True,
                    "require_tcn_agreement": True,
                }
            }
        )
    reason = apply_price_zone_gate(
        {"indicators": {"bb_pct_b": 0.2, "keltner": 0.2}, "trend_direction": "CALL", "dl_direction": "CALL"},
        TradeDirection.CALL,
        None,
    )
    assert reason is None or reason.startswith("price_zone")


def test_live_signal_metrics_resolve_reset_and_missing(monkeypatch):
    reset_live_signal_metrics_config_cache()
    base = load_live_signal_metrics_from_settings()
    got = resolve_live_signal_metrics_config({"live_signal_metrics": dict(base)})
    assert got["window"] == base["window"]
    assert resolve_live_signal_metrics_config(None)["window"] == base["window"]
    reset_live_signal_metrics_config_cache()
    assert live_mod._CACHE["live"] is None
    monkeypatch.setattr(live_mod, "repo_path", lambda *a, **k: _FakePath({"deep_learning": {}}))
    with pytest.raises(ValueError, match="live_signal_metrics"):
        load_live_signal_metrics_from_settings()
    reset_live_signal_metrics_config_cache()


def test_dl_calibration_tolerance_epochs_and_params(monkeypatch):
    monkeypatch.setattr(dl_calib_mod, "repo_path", lambda *a, **k: _FakePath({"deep_learning": {"calibration": {}}}))
    with pytest.raises(ValueError, match="obrigatorio"):
        dl_calib_mod._calib_bounds()
    monkeypatch.setattr(dl_tol_mod, "repo_path", lambda *a, **k: _FakePath({"deep_learning": {"calibration": {}}}))
    with pytest.raises(ValueError, match="obrigatorio"):
        dl_tol_mod._tol()
    monkeypatch.setattr(dl_epochs_mod, "repo_path", lambda *a, **k: _FakePath({"deep_learning": {}}))
    with pytest.raises(ValueError, match="aux_regression_weight"):
        dl_epochs_mod._aux_regression_weight()
    parsed = parse_calibration_config({"calibration": {"calibration_neutral_drift": "x", "neutral_half_width": 0.05}})
    assert parsed["calibration_neutral_drift"] == pytest.approx([0.45, 0.55])
