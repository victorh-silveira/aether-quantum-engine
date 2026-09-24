"""Testes unitarios para o servico de aplicacao de Alpha Flip Adaptativo."""

from types import SimpleNamespace

from src.application.services.direction_error_reversal import (
    _get_reversal_config,
    apply_error_reversal_to_direction,
    record_error_reversal_on_settlement,
)
from src.domain.models.trade import TradeDirection


def test_get_reversal_config():
    """Verifica extracao de configuracao com override, config do orch e fallback."""
    override = {"enabled": False, "brier_threshold": 0.50}
    assert _get_reversal_config(None, override) == override

    orch_cfg = {
        "orchestrator": {
            "execution": {"error_driven_reversal": {"enabled": True, "brier_threshold": 0.35, "min_conviction": 0.55}}
        }
    }
    orch = SimpleNamespace(config=orch_cfg)
    cfg = _get_reversal_config(orch)
    assert cfg["brier_threshold"] == 0.35

    assert _get_reversal_config(None)["enabled"] is True


def test_record_error_reversal_on_settlement():
    """Verifica ciclo de vida do armamento de Alpha Flip na liquidacao."""
    record_error_reversal_on_settlement(None, "1HZ75V", won=False, raw_prob=0.65, direction="CALL")

    orch = SimpleNamespace(_active_cycle_id=10, config={})
    record_error_reversal_on_settlement(orch, "1HZ75V", won=True, raw_prob=0.65, direction="CALL")
    assert "1HZ75V" not in getattr(orch, "_pending_alpha_reversal", {})

    record_error_reversal_on_settlement(
        orch, "1HZ75V", won=False, raw_prob=0.65, direction="CALL", cfg={"enabled": False}
    )
    assert "1HZ75V" not in getattr(orch, "_pending_alpha_reversal", {})

    record_error_reversal_on_settlement(orch, "1HZ75V", won=False, raw_prob=None, direction="CALL")
    record_error_reversal_on_settlement(orch, "1HZ75V", won=False, raw_prob=0.65, direction="INVALID")
    record_error_reversal_on_settlement(orch, "1HZ75V", won=False, raw_prob=0.52, direction="CALL")
    assert "1HZ75V" not in getattr(orch, "_pending_alpha_reversal", {})

    record_error_reversal_on_settlement(orch, "1HZ75V", won=False, raw_prob=0.65, direction="CALL")
    bag = getattr(orch, "_pending_alpha_reversal", {})
    assert "1HZ75V" in bag
    assert bag["1HZ75V"]["target_dir"] == "PUT"
    assert bag["1HZ75V"]["from_dir"] == "CALL"
    assert bag["1HZ75V"]["armed_cycle"] == 10


def test_apply_error_reversal_to_direction():
    """Verifica consumo e inversao do sinal de trade no ciclo seguinte."""
    metrics = {}
    dir_res, flipped = apply_error_reversal_to_direction(None, "1HZ75V", TradeDirection.CALL, metrics)
    assert not flipped
    assert dir_res == TradeDirection.CALL

    orch = SimpleNamespace(_pending_alpha_reversal={})
    dir_res, flipped = apply_error_reversal_to_direction(orch, "1HZ75V", TradeDirection.CALL, metrics)
    assert not flipped

    orch._pending_alpha_reversal["1HZ75V"] = {
        "brier": 0.4225,
        "residual": -0.65,
        "from_dir": "CALL",
        "target_dir": "PUT",
    }
    dir_res, flipped = apply_error_reversal_to_direction(
        orch, "1HZ75V", TradeDirection.CALL, metrics, exec_cfg={"error_driven_reversal": {"enabled": False}}
    )
    assert not flipped

    orch._pending_alpha_reversal["1HZ75V"] = {
        "brier": 0.4225,
        "residual": -0.65,
        "from_dir": "CALL",
        "target_dir": "PUT",
    }
    dir_res, flipped = apply_error_reversal_to_direction(orch, "1HZ75V", TradeDirection.CALL, metrics)
    assert flipped
    assert dir_res == TradeDirection.PUT
    assert metrics["alpha_flip_applied"] is True
    assert metrics["direction_origin"] == "FLIP_ERROR_DRIVEN_ALPHA"
    assert "1HZ75V" not in orch._pending_alpha_reversal

    orch._pending_alpha_reversal["1HZ75V"] = {
        "brier": 0.4225,
        "residual": 0.65,
        "from_dir": "PUT",
        "target_dir": "INVALID",
    }
    dir_res2, flipped2 = apply_error_reversal_to_direction(orch, "1HZ75V", TradeDirection.PUT, {})
    assert flipped2
    assert dir_res2 == TradeDirection.CALL
