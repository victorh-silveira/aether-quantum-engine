from src.application.services.llm.macro_snapshot_build import _effective_min_indices


def test_effective_min_indices_scales_with_cluster_size():
    assert _effective_min_indices(2, ["OTC_DJI"]) == 1
    assert _effective_min_indices(2, ["OTC_DJI", "OTC_NDX"]) == 2
    assert _effective_min_indices(2, []) == 2
