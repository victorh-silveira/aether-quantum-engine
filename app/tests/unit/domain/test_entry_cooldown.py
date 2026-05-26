from src.domain.risk.entry_cooldown import resolve_entry_cooldown_seconds, resolve_entry_cooldown_ticks


def test_resolve_entry_cooldown_base():
    risk = {"params": {"entry_cooldown_ticks": 12}}
    assert resolve_entry_cooldown_ticks(risk, 0.5) == 12


def test_resolve_entry_cooldown_seconds_and_high_conviction():
    risk = {
        "params": {
            "entry_cooldown_seconds": 60,
            "entry_cooldown_seconds_high_conviction": 30,
            "high_conviction_cooldown_threshold": 0.85,
        }
    }
    assert resolve_entry_cooldown_seconds(risk, 0.5) == 60.0
    assert resolve_entry_cooldown_seconds(risk, 0.9) == 30.0
    assert resolve_entry_cooldown_seconds({"params": {}}, 0.5) is None


def test_resolve_entry_cooldown_high_conviction():
    risk = {
        "params": {
            "entry_cooldown_ticks": 12,
            "entry_cooldown_ticks_high_conviction": 6,
            "high_conviction_cooldown_threshold": 0.85,
        }
    }
    assert resolve_entry_cooldown_ticks(risk, 0.9) == 6
    assert resolve_entry_cooldown_ticks(risk, 0.7) == 12
