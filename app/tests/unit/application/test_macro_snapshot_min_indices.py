from src.application.services.llm.macro_snapshot_build import _effective_min_indices


def test_effective_min_indices_scales_with_cluster_size():
    assert _effective_min_indices(2, ["R_50"]) == 1
    assert _effective_min_indices(2, ["R_50", "R_50"]) == 2
    assert _effective_min_indices(2, []) == 2
