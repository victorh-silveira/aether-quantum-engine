import pytest

from src.application.services.market_audit_ops_window import resolve_hybrid_candle_anchor


def test_resolve_hybrid_candle_anchor_both_agree():
    metrics = {
        "ops_window_candle_dir": "CALL",
        "ops_window_candle_body": 0.5,
        "ops_window_stamped": True,
        "closed_micro_candle_dir": "CALL",
        "closed_micro_candle_body": 0.8,
    }
    side, body, agree = resolve_hybrid_candle_anchor(metrics)
    assert side == "CALL"
    assert body == pytest.approx(0.8)
    assert agree is True


def test_resolve_hybrid_candle_anchor_discord():
    metrics = {
        "ops_window_candle_dir": "CALL",
        "ops_window_candle_body": 0.5,
        "ops_window_stamped": True,
        "closed_micro_candle_dir": "PUT",
        "closed_micro_candle_body": 0.3,
    }
    side, body, agree = resolve_hybrid_candle_anchor(metrics)
    assert side == "CALL"
    assert body == pytest.approx(0.3)
    assert agree is False


def test_resolve_hybrid_candle_anchor_discord_without_bodies():
    metrics = {
        "ops_window_candle_dir": "CALL",
        "ops_window_candle_body": None,
        "ops_window_stamped": True,
        "closed_micro_candle_dir": "PUT",
        "closed_micro_candle_body": None,
    }
    side, body, agree = resolve_hybrid_candle_anchor(metrics)
    assert side == "CALL"
    assert body is None
    assert agree is False


def test_resolve_hybrid_candle_anchor_ops_incomplete_fallback():
    metrics = {
        "ops_window_candle_dir": None,
        "ops_window_candle_body": None,
        "ops_window_stamped": False,
        "closed_micro_candle_dir": "PUT",
        "closed_micro_candle_body": 1.2,
    }
    side, body, agree = resolve_hybrid_candle_anchor(metrics)
    assert side == "PUT"
    assert body == pytest.approx(1.2)
    assert agree is False


def test_resolve_hybrid_candle_anchor_no_last_candle():
    metrics = {
        "ops_window_candle_dir": "CALL",
        "ops_window_candle_body": 0.5,
        "ops_window_stamped": True,
    }
    side, body, agree = resolve_hybrid_candle_anchor(metrics)
    assert side == "CALL"
    assert body == pytest.approx(0.5)
    assert agree is False


def test_resolve_hybrid_candle_anchor_none_metrics():
    assert resolve_hybrid_candle_anchor(None) == (None, None, False)
    assert resolve_hybrid_candle_anchor({}) == (None, None, False)


def test_resolve_hybrid_candle_anchor_invalid_last_body():
    metrics = {
        "ops_window_candle_dir": "CALL",
        "ops_window_candle_body": 0.5,
        "ops_window_stamped": True,
        "closed_micro_candle_dir": "CALL",
        "closed_micro_candle_body": "bad",
    }
    side, body, agree = resolve_hybrid_candle_anchor(metrics)
    assert side == "CALL"
    assert body == pytest.approx(0.5)
    assert agree is True
