"""Gates elegiveis a re-resolucao sob recovery."""

from __future__ import annotations


RECOVERY_RERESOLVE_GATES = frozenset(
    {
        "meta_shadow_inverted_veto",
        "meta_payoff_negative_zscore_veto",
        "meta_negative_edge",
        "side_imbalance_flip_not_better",
        "side_imbalance_thin_margin_flip",
        "side_imbalance_both_sides",
        "side_imbalance_flip_zone_conflict",
        "side_imbalance_large_n_margin",
    }
)
