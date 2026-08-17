from src.application.services.loss_classifier_flip import (
    closed_micro_candle_side,
    flip_reason_token,
    post_flip_edge_ok,
    resolve_flip_p_loss_floor,
    resolve_flip_waivers,
    tcn_pos_edge_blocks_flip,
)
from src.domain.models.trade import TradeDirection


def test_closed_micro_candle_side_prefers_explicit():
    assert closed_micro_candle_side({"closed_micro_candle_dir": "PUT"}) == "PUT"
    assert closed_micro_candle_side({"closed_micro_candle_dir": "PUT", "scale_micro_bar_dir": "CALL"}) == "PUT"
    assert closed_micro_candle_side({"scale_micro_bar_dir": "CALL", "scale_micro_prev_bar_dir": "PUT"}) == "CALL"
    assert closed_micro_candle_side({"scale_micro_prev_bar_dir": "CALL"}) is None
    assert closed_micro_candle_side({}) is None
    assert closed_micro_candle_side(None) is None
    assert closed_micro_candle_side("PUT") is None


def test_flip_reason_token_candle():
    assert flip_reason_token({"loss_clf_flip_candle_waive_scale": True}) == "candle"
    assert flip_reason_token({"loss_clf_flip_candle_waive_edge": True}) == "candle"
    assert flip_reason_token({"loss_clf_flip_scale_p_override": True}) == "p_ovr"


def test_tcn_pos_edge_blocks_flip():
    metrics = {"calibrated_prob": 0.36, "raw_prob": 0.36}
    cfg = {"flip_block_when_tcn_pos_edge": True, "flip_min_edge_execute": 0.04}
    assert tcn_pos_edge_blocks_flip(metrics, TradeDirection.PUT, cfg=cfg) is True
    assert metrics.get("loss_clf_flip_block_tcn_pos_edge") is True
    weak = {"calibrated_prob": 0.52, "raw_prob": 0.52}
    assert tcn_pos_edge_blocks_flip(weak, TradeDirection.CALL, cfg=cfg) is False
    assert (
        tcn_pos_edge_blocks_flip(
            {"calibrated_prob": 0.36, "raw_prob": 0.36},
            TradeDirection.PUT,
            cfg={"flip_block_when_tcn_pos_edge": False, "flip_min_edge_execute": 0.04},
        )
        is False
    )


def test_tcn_pos_edge_allows_flip_when_cal_pos_raw_neg():
    metrics = {"calibrated_prob": 0.24, "raw_prob": 0.49}
    cfg = {
        "flip_block_when_tcn_pos_edge": True,
        "flip_min_edge_execute": 0.04,
        "flip_tcn_pos_edge_raw_floor": 0.0,
    }
    assert tcn_pos_edge_blocks_flip(metrics, TradeDirection.PUT, cfg=cfg) is False
    assert metrics.get("loss_clf_flip_cal_raw_discord") is True
    assert metrics.get("loss_clf_flip_block_tcn_pos_edge") is None


def test_flip_reason_token_candle_floor():
    assert flip_reason_token({"loss_clf_flip_candle_floor": True}) == "candle"


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


def test_flip_p_loss_override_clears_seed_and_scale():
    metrics = {
        "scale_tape_consensus": "CALL",
        "scale_vote_call_n": 4,
        "scale_vote_put_n": 0,
        "closed_micro_candle_dir": "CALL",
        "calibrated_prob": 0.52,
    }
    response = {"auto_learn_applied": False, "p_loss": 0.957}
    cfg = {
        "flip_require_auto_learn": True,
        "flip_allow_seed_on_scale_discord": True,
        "flip_allow_seed_on_cal_discord": True,
        "flip_cal_discord_margin": 0.03,
        "flip_waive_on_closed_candle": True,
        "flip_waive_scale_above_p_loss": 0.95,
    }
    seed_block, scale_block = resolve_flip_waivers(metrics, response, TradeDirection.CALL, cfg=cfg, p_loss=0.957)
    assert seed_block is False
    assert scale_block is False
    assert metrics.get("loss_clf_flip_scale_p_override") is True
    assert metrics.get("loss_clf_flip_seed_p_override") is True


def test_flip_candle_lowers_p_loss_floor():
    metrics = {"closed_micro_candle_dir": "PUT", "calibrated_prob": 0.52}
    cfg = {
        "hard_p_loss_floor": 0.90,
        "flip_candle_p_loss_floor": 0.85,
        "flip_waive_on_closed_candle": True,
        "flip_min_edge_execute": 0.04,
    }
    assert resolve_flip_p_loss_floor(metrics, TradeDirection.CALL, cfg=cfg) == 0.85
    assert metrics.get("loss_clf_flip_candle_floor") is True
    assert resolve_flip_p_loss_floor({"closed_micro_candle_dir": "CALL"}, TradeDirection.CALL, cfg=cfg) == 0.90
    strong = {"closed_micro_candle_dir": "CALL", "calibrated_prob": 0.36}
    assert resolve_flip_p_loss_floor(strong, TradeDirection.PUT, cfg=cfg) == 0.90


def test_post_flip_edge_ok_waives_when_candle_agrees():
    metrics = {"calibrated_prob": 0.44, "closed_micro_candle_dir": "PUT"}
    cfg = {
        "flip_require_pos_edge": True,
        "flip_min_edge_execute": 0.04,
        "flip_waive_on_closed_candle": True,
        "flip_waive_edge_min": -1.0,
    }
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


def test_post_flip_edge_ok_rejects_deep_negative_even_with_candle():
    metrics = {"calibrated_prob": 0.36, "closed_micro_candle_dir": "CALL"}
    cfg = {
        "flip_require_pos_edge": True,
        "flip_min_edge_execute": 0.04,
        "flip_waive_on_closed_candle": True,
        "flip_waive_edge_min": -0.05,
    }
    assert post_flip_edge_ok(metrics, TradeDirection.CALL, cfg=cfg) is False


def test_post_flip_edge_ok_waives_deep_negative_with_ssot_min():
    metrics = {"calibrated_prob": 0.36, "closed_micro_candle_dir": "CALL"}
    cfg = {
        "flip_require_pos_edge": True,
        "flip_min_edge_execute": 0.04,
        "flip_waive_on_closed_candle": True,
        "flip_waive_edge_min": -1.0,
    }
    assert post_flip_edge_ok(metrics, TradeDirection.CALL, cfg=cfg) is True
    assert metrics.get("loss_clf_flip_candle_waive_edge") is True


def test_post_flip_edge_ok_waives_on_p_ovr():
    metrics = {
        "calibrated_prob": 0.56,
        "closed_micro_candle_dir": "CALL",
        "loss_clf_flip_scale_p_override": True,
    }
    cfg = {
        "flip_require_pos_edge": True,
        "flip_min_edge_execute": 0.04,
        "flip_waive_on_closed_candle": True,
        "flip_waive_edge_min": -1.0,
    }
    assert post_flip_edge_ok(metrics, TradeDirection.CALL, cfg=cfg) is True
    assert metrics.get("loss_clf_flip_p_ovr_waive_edge") is True
