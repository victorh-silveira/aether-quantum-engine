from src.application.services.loss_classifier_flip import (
    closed_micro_candle_side,
    flip_reason_token,
    post_flip_edge_ok,
    resolve_flip_waivers,
)
from src.domain.models.trade import TradeDirection


def test_closed_micro_candle_side_prefers_explicit():
    assert closed_micro_candle_side({"closed_micro_candle_dir": "PUT"}) == "PUT"
    assert closed_micro_candle_side({"scale_micro_prev_bar_dir": "CALL"}) == "CALL"
    assert closed_micro_candle_side({}) is None
    assert closed_micro_candle_side(None) is None
    assert closed_micro_candle_side("PUT") is None


def test_flip_reason_token_candle():
    assert flip_reason_token({"loss_clf_flip_candle_waive_scale": True}) == "candle"
    assert flip_reason_token({"loss_clf_flip_candle_waive_edge": True}) == "candle"


def test_flip_waives_scale_when_closed_candle_opposes_tcn():
    metrics = {
        "scale_tape_consensus": "CALL",
        "scale_vote_call_n": 4,
        "scale_vote_put_n": 0,
        "closed_micro_candle_dir": "PUT",
        "calibrated_prob": 0.57,
    }
    response = {"auto_learn_applied": True, "p_loss": 0.96}
    cfg = {
        "flip_require_auto_learn": True,
        "flip_allow_seed_on_scale_discord": True,
        "flip_allow_seed_on_cal_discord": True,
        "flip_cal_discord_margin": 0.03,
        "flip_waive_on_closed_candle": True,
    }
    seed_block, scale_block = resolve_flip_waivers(metrics, response, TradeDirection.CALL, cfg=cfg)
    assert seed_block is False
    assert scale_block is False
    assert metrics.get("loss_clf_flip_candle_waive_scale") is True


def test_post_flip_edge_ok_waives_when_candle_agrees():
    metrics = {"calibrated_prob": 0.57, "closed_micro_candle_dir": "PUT"}
    cfg = {"flip_require_pos_edge": True, "flip_min_edge_execute": 0.04, "flip_waive_on_closed_candle": True}
    assert post_flip_edge_ok(metrics, TradeDirection.PUT, cfg=cfg) is True
    assert metrics.get("loss_clf_flip_candle_waive_edge") is True
    assert (
        post_flip_edge_ok(
            {"calibrated_prob": 0.57, "closed_micro_candle_dir": "CALL"},
            TradeDirection.PUT,
            cfg=cfg,
        )
        is False
    )
