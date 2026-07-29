import json
from pathlib import Path

import pytest

from src.application.services.deep_learning.dl_calibration_tolerance import (
    apply_calibration_neutral_tolerance,
    infer_direction_from_prob,
)
from src.domain.risk.consensus_stake_penalty import apply_soft_recovery_stake, max_safe_stake_cap


def test_load_settings_json_structure():
    settings_path = Path("settings.json")
    if settings_path.exists():
        with settings_path.open(encoding="utf-8") as f:
            data = json.load(f)
        assert "anchor" in data
        assert "deep_learning" in data
        assert "risk_management" in data


@pytest.mark.parametrize(
    "raw_prob, direction",
    [
        (0.85, "CALL"),
        (0.15, "PUT"),
        (0.50, "CALL"),
        (0.20, "PUT"),
    ],
)
def test_prediction_cases_simulation(raw_prob, direction):
    res_dir = infer_direction_from_prob(raw_prob, direction)
    assert res_dir in ("CALL", "PUT", "NEUTRAL", None)

    # Coleta todos os retornos em uma tupla para evitar erro de descompactação
    result = apply_calibration_neutral_tolerance(raw_prob=raw_prob, calibrated_prob=raw_prob, direction=direction)
    assert result is not None
    assert isinstance(result, (list, tuple))
    assert 0.0 <= float(result[0]) <= 1.0


@pytest.mark.parametrize(
    "pending_loss, payout",
    [
        (15.0, 0.82),
        (50.0, 0.90),
    ],
)
def test_risk_stake_recovery_scenarios(pending_loss, payout):
    bankroll = 1000.0
    stake = apply_soft_recovery_stake(
        pending_total=pending_loss,
        payout=payout,
        bankroll=bankroll,
        consecutive_losses=2,
        base_unit=1.0,
        previous_stake=1.0,
    )
    assert stake >= 1.0
    cap = max_safe_stake_cap(bankroll, consecutive_losses_linear=2)
    assert stake <= cap
