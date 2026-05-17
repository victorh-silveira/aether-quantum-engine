from src.application.services.llm.llm_bridge_utils import strict_normalize_direction


def test_strict_normalize_direction_single_token():
    assert strict_normalize_direction("  call \n") == "CALL"
    assert strict_normalize_direction("WAIT") is None
    assert strict_normalize_direction("") is None
    assert strict_normalize_direction("sem sinal") is None


def test_strict_normalize_direction_ambiguous_call_put_is_none():
    assert strict_normalize_direction("CALL vs PUT") is None
