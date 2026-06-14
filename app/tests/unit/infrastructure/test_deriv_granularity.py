from src.infrastructure.api.deriv_granularity import normalize_granularity_seconds


def test_normalize_granularity_keeps_allowed_values():
    assert normalize_granularity_seconds(60) == 60
    assert normalize_granularity_seconds(300) == 300


def test_normalize_granularity_snaps_invalid_to_minimum():
    assert normalize_granularity_seconds(10) == 60
    assert normalize_granularity_seconds(45) == 60
    assert normalize_granularity_seconds(200) == 300
    assert normalize_granularity_seconds(100000) == 86400
