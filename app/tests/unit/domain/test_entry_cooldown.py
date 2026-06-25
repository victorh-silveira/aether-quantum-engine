from src.domain.risk.entry_cooldown import resolve_entry_cooldown_seconds, resolve_entry_cooldown_ticks


def test_resolve_entry_cooldown_always_disabled():
    risk = {
        "params": {
            "entry_cooldown_ticks": 12,
            "entry_cooldown_ticks_high_conviction": 6,
            "entry_cooldown_seconds": 60,
            "entry_cooldown_seconds_high_conviction": 30,
            "high_conviction_cooldown_threshold": 0.85,
        }
    }
    # Sempre deve retornar None para segundos e 0 para ticks, mesmo com configuracoes ativas
    assert resolve_entry_cooldown_ticks(risk, 0.5) == 0
    assert resolve_entry_cooldown_ticks(risk, 0.9) == 0
    assert resolve_entry_cooldown_seconds(risk, 0.5) is None
    assert resolve_entry_cooldown_seconds(risk, 0.9) is None
