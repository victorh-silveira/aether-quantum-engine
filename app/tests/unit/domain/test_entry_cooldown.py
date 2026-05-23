from src.domain.risk.entry_cooldown import resolve_entry_cooldown_ticks


def test_resolve_entry_cooldown_base():
    risk = {"params": {"entry_cooldown_ticks": 12}}
    assert resolve_entry_cooldown_ticks(risk, 0.5) == 12


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
