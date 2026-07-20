import pytest

from src.application.services.execution_loss_protection import (
    _directional_calibrated_side,
    _directional_raw_side,
    apply_loss_protection_penalties,
    candidate_passes_loss_protection,
    edge_conviction_disconnect_penalty,
    filter_loss_protection_candidates,
    filter_recovery_hurst_candidates,
)
from src.domain.models.trade import TradeDirection


def _candidate(**metrics):
    base = {
        "direction_margin": 0.28,
        "edge": 0.30,
        "edge_zscore": 0.50,
        "raw_prob": 0.25,
        "calibrated_prob": 0.22,
    }
    base.update(metrics)
    return ("R_10", TradeDirection.PUT, base)


def test_edge_conviction_disconnect_penalty_high_edge_low_margin():
    metrics = {"edge": 0.55, "direction_margin": 0.16, "edge_zscore": 0.40, "raw_prob": 0.40}
    assert edge_conviction_disconnect_penalty(metrics) >= 0.16


def test_apply_loss_protection_penalties_sets_metric():
    metrics = {"edge": 0.60, "direction_margin": 0.10, "edge_zscore": 1.0, "raw_prob": 0.35}
    apply_loss_protection_penalties(metrics)
    assert metrics.get("loss_protection_penalty", 0.0) > 0.0


def test_candidate_passes_loss_protection_blocks_recovery_low_margin():
    item = _candidate(direction_margin=0.15, edge=1.19)
    exec_cfg = {"loss_protection": {"recovery_min_direction_margin": 0.20}}
    assert (
        candidate_passes_loss_protection(
            item,
            exec_cfg=exec_cfg,
            recovery_active=True,
            consecutive_losses=2,
        )
        is True
    )


def test_candidate_passes_loss_protection_blocks_low_margin_high_zscore_only():
    item = _candidate(direction_margin=0.10, edge=0.25, edge_zscore=1.0)
    assert (
        candidate_passes_loss_protection(
            item,
            exec_cfg={},
            recovery_active=False,
            consecutive_losses=0,
        )
        is True
    )


def test_filter_loss_protection_candidates_keeps_fallback_pool():
    weak = _candidate(direction_margin=0.10, edge=0.95, edge_zscore=1.2)
    strong = _candidate(direction_margin=0.30, edge=0.25, edge_zscore=0.40)
    exec_cfg = {"loss_protection": {"min_direction_margin": 0.18, "max_edge_without_margin": 0.40}}
    filtered = filter_loss_protection_candidates(
        [weak, strong],
        exec_cfg=exec_cfg,
        recovery_active=False,
        consecutive_losses=0,
    )
    assert filtered == [weak, strong]


def test_filter_recovery_hurst_candidates_prefers_persistent_at_n2():
    low = ("R_10", TradeDirection.PUT, {"indicators": {"hurst": 0.52}})
    high = ("R_10", TradeDirection.PUT, {"indicators": {"hurst": 0.61}})
    filtered = filter_recovery_hurst_candidates(
        [low, high],
        kelly_cfg={"recovery_hurst_persistence_min": 0.58},
        consecutive_losses=2,
    )
    assert filtered == [low, high]


def test_edge_conviction_disconnect_penalty_weak_directional_raw():
    metrics = {"edge": 0.45, "direction_margin": 0.30, "edge_zscore": 0.2, "raw_prob": 0.80}
    assert edge_conviction_disconnect_penalty(metrics, exec_direction=TradeDirection.PUT) >= 0.15


def test_candidate_passes_loss_protection_rejects_high_penalty():
    item = _candidate(direction_margin=0.12, edge=0.55, edge_zscore=0.2, raw_prob=0.65, calibrated_prob=0.30)
    assert (
        candidate_passes_loss_protection(
            item,
            exec_cfg={},
            recovery_active=False,
            consecutive_losses=0,
        )
        is True
    )


def test_filter_loss_protection_candidates_keeps_pool_in_recovery_when_all_weak():
    weak = _candidate(direction_margin=0.10, edge=0.90, edge_zscore=1.5, indicators={"hurst": 0.42})
    filtered = filter_loss_protection_candidates(
        [weak],
        exec_cfg={"loss_protection": {"min_direction_margin": 0.18, "recovery_min_hurst": 0.50}},
        recovery_active=True,
        consecutive_losses=1,
    )
    assert filtered == [weak]


def test_candidate_passes_loss_protection_allows_missing_hurst_in_recovery():
    item = _candidate(direction_margin=0.30, edge=0.20, indicators={})
    assert (
        candidate_passes_loss_protection(
            item,
            exec_cfg={"loss_protection": {"recovery_min_hurst": 0.50}},
            recovery_active=True,
            consecutive_losses=1,
        )
        is True
    )


def test_candidate_passes_loss_protection_blocks_low_hurst_in_recovery():
    item = _candidate(direction_margin=0.30, edge=0.20, indicators={"hurst": 0.44})
    assert (
        candidate_passes_loss_protection(
            item,
            exec_cfg={"loss_protection": {"recovery_min_hurst": 0.50}},
            recovery_active=True,
            consecutive_losses=1,
        )
        is True
    )


def test_filter_loss_protection_candidates_fallback_when_all_weak_normal_mode():
    weak = _candidate(direction_margin=0.05, edge=0.90, edge_zscore=1.5)
    filtered = filter_loss_protection_candidates(
        [weak],
        exec_cfg={"loss_protection": {"min_direction_margin": 0.18}},
        recovery_active=False,
        consecutive_losses=0,
    )
    assert filtered == [weak]


def test_filter_recovery_hurst_candidates_returns_all_before_n2():
    low = ("R_10", TradeDirection.PUT, {"indicators": {"hurst": 0.52}})
    assert filter_recovery_hurst_candidates([low], kelly_cfg={}, consecutive_losses=1) == [low]


def test_filter_recovery_hurst_candidates_keeps_pool_at_n3_without_persistence():
    low = ("R_10", TradeDirection.PUT, {"indicators": {"hurst": 0.52}})
    filtered = filter_recovery_hurst_candidates(
        [low],
        kelly_cfg={"recovery_hurst_persistence_min": 0.58},
        consecutive_losses=3,
    )
    assert filtered == [low]


def test_calibrated_side_invalid_value_returns_zero():
    assert _directional_calibrated_side({"calibrated_prob": "bad"}) == 0.0


def test_directional_raw_side_invalid_and_call_put_paths():
    assert _directional_raw_side({"raw_prob": "bad"}, TradeDirection.CALL) == 0.0
    assert _directional_raw_side({"raw_prob": 0.70}, TradeDirection.CALL) == pytest.approx(0.70)
    assert _directional_raw_side({"raw_prob": 0.70}, TradeDirection.PUT) == pytest.approx(0.30)


def test_edge_conviction_disconnect_penalty_invalid_metrics():
    assert edge_conviction_disconnect_penalty({"edge": "bad"}) == 0.0


def test_edge_conviction_disconnect_penalty_high_z_low_margin():
    metrics = {"edge": 0.20, "direction_margin": 0.20, "edge_zscore": 1.0, "raw_prob": 0.60}
    assert edge_conviction_disconnect_penalty(metrics) >= 0.14


def test_edge_conviction_disconnect_penalty_low_calibrated_side():
    metrics = {
        "edge": 0.55,
        "direction_margin": 0.30,
        "edge_zscore": 0.2,
        "raw_prob": 0.60,
        "calibrated_prob": 0.20,
    }
    assert edge_conviction_disconnect_penalty(metrics, exec_direction=TradeDirection.CALL) >= 0.18


def test_apply_loss_protection_penalties_reads_exec_direction_from_metrics():
    metrics = {
        "edge": 0.55,
        "direction_margin": 0.30,
        "edge_zscore": 0.2,
        "raw_prob": 0.80,
        "exec_direction": "PUT",
    }
    apply_loss_protection_penalties(metrics)
    assert metrics.get("loss_protection_penalty", 0.0) >= 0.15


def test_candidate_passes_loss_protection_rejects_invalid_tuple():
    assert (
        candidate_passes_loss_protection(
            ("R_10",),
            exec_cfg={},
            recovery_active=False,
            consecutive_losses=0,
        )
        is False
    )


def test_candidate_passes_loss_protection_rejects_invalid_metrics():
    assert (
        candidate_passes_loss_protection(
            ("R_10", TradeDirection.PUT, "bad"),
            exec_cfg={},
            recovery_active=False,
            consecutive_losses=0,
        )
        is False
    )


def test_filter_recovery_hurst_candidates_skips_malformed_items():
    malformed = ("R_10", TradeDirection.PUT)
    good = ("R_10", TradeDirection.PUT, {"indicators": {"hurst": 0.61}})
    filtered = filter_recovery_hurst_candidates(
        [malformed, good],
        kelly_cfg={"recovery_hurst_persistence_min": 0.58},
        consecutive_losses=2,
    )
    assert filtered == [malformed, good]


def test_filter_recovery_hurst_candidates_skips_non_dict_metrics():
    bad = ("R_10", TradeDirection.PUT, "metrics")
    good = ("R_10", TradeDirection.PUT, {"indicators": {"hurst": 0.61}})
    filtered = filter_recovery_hurst_candidates(
        [bad, good],
        kelly_cfg={"recovery_hurst_persistence_min": 0.58},
        consecutive_losses=2,
    )
    assert filtered == [bad, good]


def test_filter_recovery_hurst_candidates_keeps_pool_at_n2_without_persistence():
    low = ("R_10", TradeDirection.PUT, {"indicators": {"hurst": 0.52}})
    assert filter_recovery_hurst_candidates(
        [low],
        kelly_cfg={"recovery_hurst_persistence_min": 0.58},
        consecutive_losses=2,
    ) == [low]


def test_candidate_passes_loss_protection_handles_non_dict_exec_cfg():
    item = _candidate(direction_margin=0.30, edge=0.20)
    assert (
        candidate_passes_loss_protection(
            item,
            exec_cfg="invalid",
            recovery_active=False,
            consecutive_losses=0,
        )
        is True
    )


def test_candidate_passes_loss_protection_handles_invalid_nested_cfg():
    item = _candidate(direction_margin=0.30, edge=0.20)
    assert (
        candidate_passes_loss_protection(
            item,
            exec_cfg={"loss_protection": "invalid"},
            recovery_active=False,
            consecutive_losses=0,
        )
        is True
    )
