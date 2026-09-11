"""Testes do adapt SCALE por retracao/explosao (ultima palavra apos FLIP; sem SKIP)."""

from __future__ import annotations

from src.application.services.execution_scale_adapt import apply_scale_retract_adapt
from src.domain.models.trade import TradeDirection


_CFG = {
    "adapt_retract_enabled": True,
    "retraction_require_mili": True,
    "adapt_tape_require_strong": True,
}


def test_adapt_c1_retract_soft_put_to_call():
    metrics = {
        "tcn_direction": "PUT",
        "direction_margin": 0.004,
        "scale_micro_regime": "retraction",
        "scale_micro_side": "CALL",
        "scale_mini_bar_dir": "CALL",
        "scale_mili_dir": "CALL",
    }
    out = apply_scale_retract_adapt(metrics, TradeDirection.PUT, cfg=_CFG)
    assert out is TradeDirection.CALL
    assert metrics["scale_adapted"] is True
    assert metrics["scale_adapt_reason"] == "retract_vs_tcn"
    assert metrics["scale_adapt_undid_flip"] is False
    assert metrics["exec_direction"] == "CALL"


def test_adapt_firm_margin_still_adapts_on_retract():
    metrics = {
        "direction_margin": 0.064,
        "scale_micro_regime": "retraction",
        "scale_mini_bar_dir": "CALL",
        "scale_mili_dir": "CALL",
    }
    out = apply_scale_retract_adapt(metrics, TradeDirection.PUT, cfg=_CFG)
    assert out is TradeDirection.CALL
    assert metrics["scale_adapted"] is True
    assert metrics["scale_adapt_reason"] == "retract_vs_tcn"


def test_adapt_c4_explos_requires_mili_alignment():
    metrics = {
        "tcn_direction": "CALL",
        "exec_direction": "CALL",
        "scale_micro_regime": "explosion",
        "scale_micro_side": "PUT",
        "scale_mini_prev_bar_dir": "PUT",
        "scale_mini_bar_dir": "PUT",
        "scale_mili_dir": "CALL",
        "scale_tape_consensus": "PUT",
    }
    out = apply_scale_retract_adapt(metrics, TradeDirection.CALL, cfg=_CFG)
    assert out is TradeDirection.CALL
    assert metrics["scale_adapted"] is False
    assert metrics["scale_adapt_reason"] == "mili_mismatch"


def test_adapt_explos_with_mili_aligned():
    metrics = {
        "tcn_direction": "CALL",
        "exec_direction": "CALL",
        "scale_micro_regime": "explosion",
        "scale_micro_side": "PUT",
        "scale_mini_bar_dir": "PUT",
        "scale_mili_dir": "PUT",
    }
    out = apply_scale_retract_adapt(metrics, TradeDirection.CALL, cfg=_CFG)
    assert out is TradeDirection.PUT
    assert metrics["scale_adapted"] is True
    assert metrics["scale_adapt_reason"] == "explos_vs_tcn"


def test_adapt_c3_tape_discord_alone_does_not_adapt():
    metrics = {
        "tcn_direction": "CALL",
        "exec_direction": "CALL",
        "scale_micro_regime": "chop",
        "scale_tape_consensus": "PUT",
        "scale_tape_strong": False,
        "scale_discordance": True,
        "scale_mini_prev_bar_dir": "PUT",
        "scale_mini_bar_dir": "CALL",
        "scale_mili_dir": "PUT",
    }
    out = apply_scale_retract_adapt(metrics, TradeDirection.CALL, cfg=_CFG)
    assert out is TradeDirection.CALL
    assert metrics["scale_adapted"] is False
    assert metrics["scale_adapt_reason"] == "tape_not_strong"


def test_adapt_tape_strong_without_discord():
    metrics = {
        "tcn_direction": "CALL",
        "exec_direction": "CALL",
        "scale_micro_regime": "chop",
        "scale_tape_consensus": "PUT",
        "scale_tape_strong": True,
        "scale_discordance": False,
    }
    out = apply_scale_retract_adapt(metrics, TradeDirection.CALL, cfg=_CFG)
    assert out is TradeDirection.PUT
    assert metrics["scale_adapt_reason"] == "tape_vs_tcn"


def test_adapt_tape_discord_when_require_strong_off():
    metrics = {
        "tcn_direction": "CALL",
        "exec_direction": "CALL",
        "scale_micro_regime": "chop",
        "scale_tape_consensus": "PUT",
        "scale_tape_strong": False,
        "scale_discordance": True,
    }
    cfg = {**_CFG, "adapt_tape_require_strong": False}
    out = apply_scale_retract_adapt(metrics, TradeDirection.CALL, cfg=cfg)
    assert out is TradeDirection.PUT
    assert metrics["scale_adapt_reason"] == "tape_vs_tcn"


def test_adapt_tape_weak_when_require_strong_off():
    metrics = {
        "tcn_direction": "CALL",
        "exec_direction": "CALL",
        "scale_micro_regime": "chop",
        "scale_tape_consensus": "PUT",
        "scale_tape_strong": False,
        "scale_discordance": False,
    }
    cfg = {**_CFG, "adapt_tape_require_strong": False}
    out = apply_scale_retract_adapt(metrics, TradeDirection.CALL, cfg=cfg)
    assert out is TradeDirection.CALL
    assert metrics["scale_adapted"] is False
    assert metrics["scale_adapt_reason"] == "not_adapt_regime"


def test_adapt_chop_without_tape_signal_stays():
    metrics = {
        "tcn_direction": "CALL",
        "exec_direction": "CALL",
        "scale_micro_regime": "chop",
        "scale_tape_consensus": "PUT",
        "scale_tape_strong": False,
        "scale_discordance": False,
    }
    out = apply_scale_retract_adapt(metrics, TradeDirection.CALL, cfg=_CFG)
    assert out is TradeDirection.CALL
    assert metrics["scale_adapted"] is False
    assert metrics["scale_adapt_reason"] == "tape_not_strong"


def test_adapt_explos_holds_undoes_flip():
    metrics = {
        "tcn_direction": "CALL",
        "exec_direction": "PUT",
        "loss_clf_flip": True,
        "scale_micro_regime": "explosion",
        "scale_micro_side": "CALL",
        "scale_mini_bar_dir": "CALL",
        "scale_mili_dir": "CALL",
    }
    out = apply_scale_retract_adapt(metrics, TradeDirection.CALL, cfg=_CFG)
    assert out is TradeDirection.CALL
    assert metrics["scale_adapted"] is True
    assert metrics["scale_adapt_reason"] == "explos_holds"
    assert metrics["scale_adapt_undid_flip"] is True


def test_adapt_skips_chop_only():
    metrics = {
        "direction_margin": 0.004,
        "scale_micro_regime": "chop",
        "scale_mini_bar_dir": "CALL",
        "scale_mili_dir": "CALL",
    }
    out = apply_scale_retract_adapt(metrics, TradeDirection.PUT, cfg=_CFG)
    assert out is TradeDirection.PUT
    assert metrics["scale_adapt_reason"] == "not_adapt_regime"


def test_adapt_requires_mili_alignment():
    metrics = {
        "direction_margin": 0.004,
        "scale_micro_regime": "retraction",
        "scale_mini_bar_dir": "CALL",
        "scale_mili_dir": "PUT",
    }
    out = apply_scale_retract_adapt(metrics, TradeDirection.PUT, cfg=_CFG)
    assert out is TradeDirection.PUT
    assert metrics["scale_adapt_reason"] == "mili_mismatch"


def test_adapt_disabled():
    metrics = {
        "direction_margin": 0.004,
        "exec_direction": "PUT",
        "scale_micro_regime": "explosion",
        "scale_mini_bar_dir": "CALL",
        "scale_mili_dir": "CALL",
    }
    cfg = {**_CFG, "adapt_retract_enabled": False}
    out = apply_scale_retract_adapt(metrics, TradeDirection.PUT, cfg=cfg)
    assert out is TradeDirection.PUT
    assert metrics["scale_adapted"] is False


def test_adapt_aligned_no_flip():
    metrics = {
        "direction_margin": 0.004,
        "scale_micro_regime": "retraction",
        "scale_mini_bar_dir": "PUT",
        "scale_mili_dir": "PUT",
    }
    out = apply_scale_retract_adapt(metrics, TradeDirection.PUT, cfg=_CFG)
    assert out is TradeDirection.PUT
    assert metrics["scale_adapt_reason"] == "aligned"


def test_adapt_no_tcn_defaults_call():
    metrics = {"scale_micro_regime": "retraction"}
    out = apply_scale_retract_adapt(metrics, "SKIP", cfg=_CFG)
    assert out is TradeDirection.CALL
    assert metrics["scale_adapt_reason"] == "no_tcn"


def test_adapt_no_target():
    metrics = {
        "direction_margin": 0.004,
        "scale_micro_regime": "retraction",
        "scale_mili_dir": "CALL",
    }
    out = apply_scale_retract_adapt(metrics, TradeDirection.PUT, cfg=_CFG)
    assert out is TradeDirection.PUT
    assert metrics["scale_adapt_reason"] == "no_target"


def test_adapt_uses_mini_dir_fallback():
    metrics = {
        "scale_micro_regime": "retraction",
        "scale_mini_dir": "CALL",
        "scale_mili_dir": "CALL",
    }
    out = apply_scale_retract_adapt(metrics, TradeDirection.PUT, cfg=_CFG)
    assert out is TradeDirection.CALL
    assert metrics["scale_adapted"] is True


def test_adapt_without_require_mili():
    metrics = {
        "direction_margin": 0.004,
        "scale_micro_regime": "retraction",
        "scale_mini_bar_dir": "CALL",
    }
    cfg = {**_CFG, "retraction_require_mili": False}
    out = apply_scale_retract_adapt(metrics, TradeDirection.PUT, cfg=cfg)
    assert out is TradeDirection.CALL


def test_adapt_c10_retract_holds_undoes_loss_clf_flip():
    metrics = {
        "tcn_direction": "CALL",
        "exec_direction": "PUT",
        "resolved_direction": "PUT",
        "loss_clf_flip": True,
        "scale_micro_regime": "retraction",
        "scale_micro_side": "CALL",
        "scale_mini_bar_dir": "CALL",
        "scale_mili_dir": "CALL",
    }
    out = apply_scale_retract_adapt(metrics, TradeDirection.CALL, cfg=_CFG)
    assert out is TradeDirection.CALL
    assert metrics["scale_adapted"] is True
    assert metrics["scale_adapt_reason"] == "retract_holds"
    assert metrics["scale_adapt_undid_flip"] is True
    assert metrics["scale_adapt_from"] == "PUT"
    assert metrics["exec_direction"] == "CALL"


def test_adapt_preserves_post_flip_when_not_retract():
    metrics = {
        "tcn_direction": "CALL",
        "exec_direction": "PUT",
        "loss_clf_flip": True,
        "scale_micro_regime": "chop",
        "scale_mini_bar_dir": "CALL",
        "scale_mili_dir": "CALL",
    }
    out = apply_scale_retract_adapt(metrics, TradeDirection.CALL, cfg=_CFG)
    assert out is TradeDirection.PUT
    assert metrics["scale_adapted"] is False
    assert metrics["scale_adapt_reason"] == "not_adapt_regime"
