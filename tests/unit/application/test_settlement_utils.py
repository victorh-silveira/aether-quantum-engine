from src.application.services.orchestrator.settlement_utils import (
    calculate_cluster_grace_period,
    min_elapsed_before_stagnant_polls,
    prune_orphan_contract_ids,
)


def test_calculate_cluster_grace_period_empty():
    assert calculate_cluster_grace_period({}, {}, 1000) == 0.0


def test_calculate_cluster_grace_period_no_expiry_fallback():
    # Caso onde contratos existem mas não têm expiry_time (ex: mock incompleto)
    assert calculate_cluster_grace_period({1: object()}, {}, 1000) == 0.0


def test_calculate_cluster_grace_period_valid():
    class MockContract:
        def __init__(self, expiry):
            self.expiry_time = expiry

    contracts = {
        1: MockContract(1100),
        2: MockContract(1200),  # Longest
    }
    # (1200 - 1000) + 45 (default slack) = 245.0
    assert calculate_cluster_grace_period(contracts, {}, 1000) == 245.0


def test_calculate_cluster_grace_period_with_slack_override():
    class MockContract:
        def __init__(self, expiry):
            self.expiry_time = expiry

    contracts = {1: MockContract(1100)}
    cfg = {"settlement_post_expiry_slack_seconds": 10.0}
    # (1100 - 1000) + 10 = 110.0
    assert calculate_cluster_grace_period(contracts, cfg, 1000) == 110.0


def test_prune_orphan_contract_ids_splits_kept_and_orphan():
    kept, orphan = prune_orphan_contract_ids([1, 2, 3], {1: object(), 3: object()})
    assert kept == [1, 3]
    assert orphan == [2]


def test_prune_orphan_contract_ids_when_no_orphans():
    kept, orphan = prune_orphan_contract_ids([10], {10: object()})
    assert kept == [10]
    assert orphan == []


def test_min_elapsed_m1_default():
    g = min_elapsed_before_stagnant_polls(
        {"duration": 1, "duration_unit": "m"}, {"settlement_post_expiry_slack_seconds": 20.0}
    )
    assert g == 80.0


def test_min_elapsed_override():
    g = min_elapsed_before_stagnant_polls(
        {"duration": 99, "duration_unit": "m"}, {"settlement_stagnant_grace_seconds": 3.5}
    )
    assert g == 3.5


def test_min_elapsed_ticks():
    g = min_elapsed_before_stagnant_polls(
        {"duration": 10, "duration_unit": "t"},
        {"settlement_tick_seconds_estimate": 2.0, "settlement_post_expiry_slack_seconds": 5.0},
    )
    assert g == 25.0


def test_min_elapsed_seconds_unit():
    g = min_elapsed_before_stagnant_polls(
        {"duration": 15, "duration_unit": "s"}, {"settlement_post_expiry_slack_seconds": 5.0}
    )
    assert g == 20.0


def test_min_elapsed_unknown_unit_falls_back_to_minutes():
    g = min_elapsed_before_stagnant_polls(
        {"duration": 2, "duration_unit": "x"}, {"settlement_post_expiry_slack_seconds": 10.0}
    )
    assert g == 130.0


def test_min_elapsed_multiplier():
    g = min_elapsed_before_stagnant_polls({"duration": "MULT"}, {})
    assert g == 3600.0
