from types import SimpleNamespace

from src.application.services.deep_learning.dl_post_loss import register_post_loss_ban
from src.application.services.execution_symbols import (
    _calib_gap_penalty,
    _trade_score,
    candidate_execution_score,
    filter_execution_candidates,
    filter_post_loss_banned_candidates,
    format_execution_alternates,
    pending_recovery_active,
    select_best_execution_candidate,
    select_mandatory_execution_candidate,
    symbols_eligible_for_execution,
)
from src.domain.models.trade import TradeDirection
from tests.market_symbols import ANCHOR, PAIR


_SELECTION = {
    "min_conviction_execute": 0.58,
    "min_edge_margin": 0.06,
    "min_val_accuracy": 0.50,
    "strong_raw": 0.65,
    "strong_edge": 0.12,
}


def test_symbols_eligible_for_execution():
    symbols = [ANCHOR, PAIR]
    assert symbols_eligible_for_execution(ANCHOR, symbols, include_anchor=False) == [PAIR]
    assert symbols_eligible_for_execution(ANCHOR, symbols, include_anchor=True) == symbols


def test_format_execution_alternates_excludes_selected():
    candidates = [
        (ANCHOR, TradeDirection.PUT, {"trade_score": 0.57, "conviction": 0.57, "val_accuracy": 0.60}),
        (PAIR, TradeDirection.CALL, {"trade_score": 0.52, "conviction": 0.52, "val_accuracy": 0.50}),
    ]
    assert format_execution_alternates(candidates, exclude_symbol=ANCHOR) == f"{PAIR}(0.52)"


def test_pending_recovery_active():
    assert pending_recovery_active({}) is False
    assert pending_recovery_active({ANCHOR: 100.0}) is True


def test_filter_execution_candidates_requires_val_or_strong_signal():
    weak = (PAIR, TradeDirection.CALL, {"trade_score": 0.54, "val_accuracy": 0.33, "edge": 0.04})
    solid = (PAIR, TradeDirection.CALL, {"trade_score": 0.59, "val_accuracy": 0.50, "edge": 0.09})
    strong = (
        ANCHOR,
        TradeDirection.PUT,
        {"trade_score": 0.55, "raw_prob": 0.66, "val_accuracy": 0.35, "edge": 0.16},
    )
    filtered = filter_execution_candidates([weak, solid, strong], selection=_SELECTION)
    symbols = {item[0] for item in filtered}
    assert symbols == {ANCHOR, PAIR}


def test_select_best_execution_candidate_diversify_margin_picks_alt():
    candidates = [
        (ANCHOR, TradeDirection.PUT, {"trade_score": 0.70, "val_accuracy": 0.50, "edge": 0.20}),
        (PAIR, TradeDirection.CALL, {"trade_score": 0.69, "val_accuracy": 0.48, "edge": 0.19}),
    ]
    best = select_best_execution_candidate(
        candidates,
        last_loss_symbol=ANCHOR,
        diversify_margin=0.05,
        recovery_active=False,
    )
    assert best[0] == PAIR


def test_select_best_execution_candidate_diversifies_last_loss_symbol():
    candidates = [
        (ANCHOR, TradeDirection.PUT, {"trade_score": 0.70, "val_accuracy": 0.50, "edge": 0.20}),
        (PAIR, TradeDirection.CALL, {"trade_score": 0.69, "val_accuracy": 0.48, "edge": 0.19}),
    ]
    best = select_best_execution_candidate(
        candidates,
        last_loss_symbol=ANCHOR,
        diversify_margin=0.10,
        recovery_active=True,
    )
    assert best[0] != ANCHOR


def test_select_best_candidate_prefers_high_val_in_recovery():
    candidates = [
        (PAIR, TradeDirection.PUT, {"trade_score": 0.57, "val_accuracy": 0.60, "edge": 0.07}),
        (
            ANCHOR,
            TradeDirection.CALL,
            {"trade_score": 0.59, "val_accuracy": 0.42, "edge": 0.09},
        ),
    ]
    best = select_best_execution_candidate(
        candidates,
        last_loss_symbol=None,
        diversify_margin=0.10,
        recovery_active=True,
    )
    assert best[0] == PAIR


def test_trade_score_falls_back_to_conviction():
    assert _trade_score({"conviction": 0.61}) == 0.61


def test_filter_keeps_strong_raw_when_below_conviction_threshold():
    strong_only = (
        ANCHOR,
        TradeDirection.PUT,
        {"conviction": 0.52, "raw_prob": 0.68, "val_accuracy": 0.35, "edge": 0.18},
    )
    filtered = filter_execution_candidates([strong_only], selection=_SELECTION)
    assert len(filtered) == 1


def test_calib_gap_penalty_reduces_ranking():
    cfg = {"max_calib_gap": 0.10}
    metrics = {"trade_score": 0.85, "raw_prob": 0.55}
    assert _calib_gap_penalty(metrics, cfg) > 0.0
    assert _calib_gap_penalty({"trade_score": 0.6}, cfg) == 0.0
    small_gap = {"trade_score": 0.62, "raw_prob": 0.58}
    assert _calib_gap_penalty(small_gap, {"max_calib_gap": 0.18}) == 0.0


def test_candidate_execution_score_recovery_weights_val_accuracy():
    normal = candidate_execution_score({"trade_score": 0.55, "val_accuracy": 0.40, "edge": 0.05}, recovery_active=False)
    recovery = candidate_execution_score(
        {"trade_score": 0.55, "val_accuracy": 0.40, "edge": 0.05}, recovery_active=True
    )
    high_val = candidate_execution_score(
        {"trade_score": 0.55, "val_accuracy": 0.60, "edge": 0.05}, recovery_active=True
    )
    assert high_val > recovery
    assert recovery != normal


def test_filter_post_loss_banned_candidates():
    orch = SimpleNamespace(config={"deep_learning": {"post_loss_flip_raw_min": 0.62}})
    register_post_loss_ban(orch, ANCHOR, TradeDirection.CALL, candle_cycles=2)
    candidates = [
        (ANCHOR, TradeDirection.CALL, {"raw_prob": 0.56}),
        (PAIR, TradeDirection.PUT, {"raw_prob": 0.43}),
    ]
    kept = filter_post_loss_banned_candidates(orch, candidates, flip_raw_min=0.62)
    assert len(kept) == 1
    assert kept[0][0] == PAIR


def test_select_mandatory_falls_back_when_all_banned():
    orch = SimpleNamespace(config={"deep_learning": {"post_loss_flip_raw_min": 0.99}})
    register_post_loss_ban(orch, ANCHOR, TradeDirection.CALL, candle_cycles=2)
    register_post_loss_ban(orch, PAIR, TradeDirection.PUT, candle_cycles=2)
    candidates = [
        (ANCHOR, TradeDirection.CALL, {"trade_score": 0.71, "execute": True, "raw_prob": 0.56}),
        (PAIR, TradeDirection.PUT, {"trade_score": 0.43, "execute": True, "raw_prob": 0.43}),
    ]
    best = select_mandatory_execution_candidate(
        orch,
        candidates,
        last_loss_symbol=None,
        diversify_margin=0.08,
        recovery_active=False,
        flip_raw_min=0.99,
    )
    assert best[0] == ANCHOR


def test_select_mandatory_skips_post_loss_banned_symbol():
    orch = SimpleNamespace(config={"deep_learning": {"post_loss_flip_raw_min": 0.62}})
    register_post_loss_ban(orch, ANCHOR, TradeDirection.CALL, candle_cycles=2)
    candidates = [
        (ANCHOR, TradeDirection.CALL, {"trade_score": 0.71, "execute": False, "raw_prob": 0.56}),
        (PAIR, TradeDirection.PUT, {"trade_score": 0.43, "execute": True, "raw_prob": 0.43}),
    ]
    best = select_mandatory_execution_candidate(
        orch,
        candidates,
        last_loss_symbol=ANCHOR,
        diversify_margin=0.08,
        recovery_active=True,
        flip_raw_min=0.62,
    )
    assert best[0] == PAIR


def test_select_mandatory_prefers_execute_true_when_available():
    orch = SimpleNamespace(config={"deep_learning": {"post_loss_flip_raw_min": 0.62}})
    candidates = [
        (ANCHOR, TradeDirection.CALL, {"trade_score": 0.71, "execute": False, "raw_prob": 0.56}),
        (PAIR, TradeDirection.PUT, {"trade_score": 0.55, "execute": True, "raw_prob": 0.55}),
    ]
    best = select_mandatory_execution_candidate(
        orch,
        candidates,
        last_loss_symbol=None,
        diversify_margin=0.08,
        recovery_active=False,
        flip_raw_min=0.62,
    )
    assert best[0] == PAIR
