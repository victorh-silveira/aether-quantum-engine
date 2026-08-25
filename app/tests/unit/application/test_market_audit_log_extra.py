from types import SimpleNamespace

import pytest

from src.application.services.market_audit_log import (
    format_cluster_audit_line,
    format_indicators_audit_line,
    pop_contract_audit,
    resolve_edge_breakeven_p,
    resolve_meta_payoff_zscore,
    resolve_predicted_edge,
    resolve_raw_predicted_edge,
    resolve_stake_audit_context,
    store_contract_audit,
)


class TestResolvePredictedEdge:
    def test_returns_edge_from_calibrated_prob(self):
        edge = resolve_predicted_edge({"calibrated_prob": 0.7}, payout=0.95)
        assert edge == pytest.approx((0.7 * 1.95) - 1.0)

    def test_falls_back_to_raw_prob(self):
        edge = resolve_predicted_edge({"raw_prob": 0.65}, payout=0.95)
        assert edge == pytest.approx((0.65 * 1.95) - 1.0)

    def test_defaults_to_0_5_when_no_prob_key(self):
        edge = resolve_predicted_edge({"some_key": 1.0}, payout=0.95)
        assert edge == pytest.approx((0.5 * 1.95) - 1.0)

    def test_returns_0_for_non_dict_input(self):
        assert resolve_predicted_edge(None) == 0.0
        assert resolve_predicted_edge("invalid") == 0.0

    def test_returns_0_when_prob_is_none(self):
        edge = resolve_predicted_edge({"calibrated_prob": None, "raw_prob": None})
        assert edge == 0.0

    def test_uses_dominant_prob_when_below_0_5(self):
        edge = resolve_predicted_edge({"raw_prob": 0.3}, payout=0.95)
        assert edge == pytest.approx((0.7 * 1.95) - 1.0)

    def test_accepts_custom_payout(self):
        edge = resolve_predicted_edge({"calibrated_prob": 0.8}, payout=0.90)
        assert edge == pytest.approx((0.8 * 1.90) - 1.0)

    def test_defaults_to_kelly_payout_fallback(self):
        edge = resolve_predicted_edge({"calibrated_prob": 0.7})
        assert edge == pytest.approx((0.7 * 1.85) - 1.0)

    def test_cal_0533_b072_identity_negative(self):
        edge = resolve_predicted_edge({"calibrated_prob": 0.533}, payout=0.72)
        assert edge == pytest.approx(-0.08324, abs=1e-5)
        assert edge < 0.0

    def test_cal_0605_b072_meets_min_edge_floor(self):
        edge = resolve_predicted_edge({"calibrated_prob": 0.605}, payout=0.72)
        assert edge == pytest.approx((0.605 * 1.72) - 1.0)
        assert edge >= 0.04

    def test_raw_edge_gap_vs_cal_edge(self):
        metrics = {"calibrated_prob": 0.53, "raw_prob": 0.99}
        cal_edge = resolve_predicted_edge(metrics, direction="CALL", payout=0.72)
        raw_edge = resolve_raw_predicted_edge(metrics, direction="CALL", payout=0.72)
        assert cal_edge < 0.0
        assert raw_edge > 0.0
        assert raw_edge == pytest.approx((0.99 * 1.72) - 1.0)
        assert resolve_edge_breakeven_p(0.72) == pytest.approx(1.0 / 1.72)

    def test_raw_edge_missing_returns_zero(self):
        assert resolve_raw_predicted_edge({}, direction="CALL", payout=0.72) == 0.0
        assert resolve_raw_predicted_edge(None) == 0.0  # type: ignore[arg-type]
        assert resolve_raw_predicted_edge({"raw_prob": "x"}, payout=0.72) == 0.0


class TestResolveMetaPayoffZscore:
    def test_returns_0_for_none(self):
        assert resolve_meta_payoff_zscore(None) == 0.0

    def test_returns_0_for_non_dict(self):
        assert resolve_meta_payoff_zscore("bad") == 0.0

    def test_reads_meta_payoff_edge_zscore(self):
        assert resolve_meta_payoff_zscore({"meta_payoff_edge_zscore": 1.5}) == 1.5

    def test_falls_back_to_edge_zscore(self):
        assert resolve_meta_payoff_zscore({"edge_zscore": -0.75}) == -0.75

    def test_returns_0_when_missing(self):
        assert resolve_meta_payoff_zscore({}) == 0.0

    def test_reads_float_string_value(self):
        assert resolve_meta_payoff_zscore({"meta_payoff_edge_zscore": "0.42"}) == 0.42


class TestStoreAndPopContractAudit:
    def test_store_and_pop_roundtrip(self):
        cid = "999"
        data = {
            "symbol": "R_10",
            "direction": "CALL",
            "edge": 0.11,
            "meta_payoff_edge_zscore": -0.55,
            "raw_prob": 0.71,
        }
        store_contract_audit(cid, data)
        assert pop_contract_audit(cid) == data

    def test_pop_returns_empty_dict_for_missing(self):
        assert pop_contract_audit("nonexistent") == {}

    def test_pop_only_once(self):
        cid = "once"
        store_contract_audit(cid, {"edge": 0.5})
        assert pop_contract_audit(cid) == {"edge": 0.5}
        assert pop_contract_audit(cid) == {}

    def test_store_ignores_empty_id(self):
        store_contract_audit("", {"edge": 0.5})

    def test_store_legacy_no_kwargs(self):
        store_contract_audit("ignored", "cid", symbol="R_10")

    def test_store_legacy_positional_no_kwargs(self):
        store_contract_audit("x", "cid")

    def test_store_legacy_too_few_args(self):
        store_contract_audit("x")

    def test_pop_fallback_empty_args(self):
        assert pop_contract_audit() == {}

    def test_store_legacy_too_few_args_kwargs(self):
        store_contract_audit(42, symbol="R_10")

    def test_store_legacy_empty_contract_id(self):
        store_contract_audit("x", "")

    def test_indicator_snapshot_non_dict(self):
        from src.application.services.market_audit_log_helpers import indicator_snapshot

        assert indicator_snapshot(None) == {}

    def test_metric_float_non_dict(self):
        from src.application.services.market_audit_log_helpers import metric_float

        assert metric_float(None, "trade_score", default=0.5) == 0.5


def test_format_indicators_audit_line():
    metrics = {
        "indicators": {
            "rsi": 0.4859,
            "adx": 0.2017,
            "hurst": 0.5671,
            "atr_norm": -0.9558,
            "bb_width": -0.2226,
            "vol_ratio": 1.0720,
        },
        "edge_zscore": 0.60,
        "val_accuracy": 0.6433,
        "direction_margin": 0.12,
        "calibration_mode": "calibrated",
        "meta_veto_mode": "none",
    }
    line = format_indicators_audit_line(6, "R_10", metrics)
    assert line.startswith("[IND] || ")
    assert "RSI:" in line and "0.4859" in line
    assert "ADX:" in line and "0.2017" in line
    assert "HURST:" in line and "0.5671" in line
    assert "ATR:" in line and "-0.9558" in line
    assert "BBW:" in line and "-0.2226" in line
    assert "VOL_R:" in line and "1.0720" in line
    assert "Z:" in line and "+0.60" in line
    assert "ACC:" in line and "0.6433" in line
    assert "MARGIN:" in line and "0.120" in line
    assert "CAL_EDGE:" in line
    assert "NEUTRAL: calibrated" in line
    assert "META_VETO: none" in line
    assert "SCALE: tcn=" in line
    assert "adapted=0" in line
    assert line.count("\n") == 3
    assert all(part.startswith("[IND] ||") for part in line.splitlines())


def test_format_indicators_audit_line_ignores_none_and_invalid():
    metrics = {"indicators": {"rsi": None, "hurst": 0.61, "adx": "bad"}, "val_accuracy": 0.5}
    line = format_indicators_audit_line(5, "R_10", metrics)
    assert "0.6100" in line
    assert "RSI:" in line
    assert "0.0000" in line


def test_format_indicators_audit_line_marks_neutral_clamp():
    metrics = {
        "indicators": {"rsi": 0.5, "adx": 0.2, "hurst": 0.5},
        "direction_margin": 0.01,
        "gate_reason": "neutral_clamp",
        "meta_veto_mode": "soft",
    }
    line = format_indicators_audit_line(7, "R_10", metrics)
    assert "NEUTRAL: neutral_clamp" in line
    assert "META_VETO: soft" in line


def test_resolve_stake_audit_context_from_audit():
    rm = SimpleNamespace(
        _last_stake_audit={
            "mode_tag": "RECOVER_DAL_L1",
            "pending": 1.5,
            "bankroll": 90.0,
            "linear_losses": 1,
            "cap": 4.2,
            "recovery_infeasible": False,
        },
        pending_loss_total=lambda: 9.0,
        bankroll=80.0,
    )
    audit = resolve_stake_audit_context(rm)
    assert audit["mode_tag"] == "RECOVER_DAL_L1"
    assert audit["pending"] == 9.0
    assert audit["bankroll"] == 90.0
    assert audit["linear"] == 1
    assert audit["cap"] == 4.2


def test_resolve_stake_audit_context_pending_after_win_abates():
    pending = {"R_10": 50.0}

    def _total() -> float:
        return float(sum(pending.values()))

    rm = SimpleNamespace(
        _last_stake_audit={
            "mode_tag": "RECOVER_DAL_L1",
            "pending": 50.0,
            "bankroll": 1000.0,
            "linear_losses": 1,
            "cap": 50.0,
            "recovery_infeasible": False,
        },
        pending_loss_total=_total,
        bankroll=1000.0,
    )
    before = resolve_stake_audit_context(rm)
    assert before["pending"] == pytest.approx(50.0)
    pending["R_10"] = max(0.0, pending["R_10"] - 12.0)
    after = resolve_stake_audit_context(rm)
    assert after["pending"] == pytest.approx(38.0)
    assert after["mode_tag"] == "RECOVER_DAL_L1"


def test_resolve_stake_audit_context_fallback_balance():
    rm = SimpleNamespace(
        bankroll=70.0,
        initial_bankroll=70.0,
        pending_loss_total=lambda: 2.0,
        consecutive_losses_linear=0,
    )
    audit = resolve_stake_audit_context(rm, balance_fallback=88.5)
    assert audit["mode_tag"] == "EXPLORE_KELLY"
    assert audit["pending"] == 2.0
    assert audit["bankroll"] == 88.5


def test_format_cluster_audit_line_empty():
    assert format_cluster_audit_line({}, timeframe="M5") == "[CLUSTER] || M5 || EMPTY"
