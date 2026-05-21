from src.application.services.llm.global_macro_confluence import (
    build_macro_snapshot,
    reconcile_cluster_tags_with_macro,
)


def test_reconcile_unknown_tag_returns_unchanged():
    snap = build_macro_snapshot(
        ["OTC_SPC"],
        ["OTC_FCHI"],
        {"OTC_SPC": [100.0, 105.0], "OTC_FCHI": [100.0, 105.0]},
        {"min_indices_for_vote": 1},
    )
    snap = type(snap)(
        us_dir="up",
        eu_dir="up",
        us_strength=1.0,
        eu_strength=1.0,
        tag="custom_unknown",
        eurusd_bias=snap.eurusd_bias,
        cluster_status=snap.cluster_status,
        macro_block=snap.macro_block,
        fx_reference_line=snap.fx_reference_line,
        us_parts=snap.us_parts,
        eu_parts=snap.eu_parts,
    )
    us, eu, changed, note = reconcile_cluster_tags_with_macro("PUT", "CALL", snap)
    assert changed is False
    assert us == "PUT"
    assert note == ""


def test_reconcile_risk_off_omits_us_when_quant_flat():
    snap = build_macro_snapshot(
        ["OTC_SPC", "OTC_NDX"],
        ["OTC_FCHI", "OTC_GDAXI"],
        {
            "OTC_SPC": [100.0, 95.0],
            "OTC_NDX": [100.0, 94.0],
            "OTC_FCHI": [100.0, 93.0],
            "OTC_GDAXI": [100.0, 92.5],
        },
        {"min_indices_for_vote": 2, "cluster_min_move_pct": 0.10},
    )
    snap = type(snap)(
        us_dir="flat",
        eu_dir="down",
        us_strength=0.66,
        eu_strength=1.0,
        tag="risk_off",
        eurusd_bias=snap.eurusd_bias,
        cluster_status=snap.cluster_status,
        macro_block=snap.macro_block,
        fx_reference_line=snap.fx_reference_line,
        us_parts=snap.us_parts,
        eu_parts=snap.eu_parts,
    )
    us, eu, changed, note = reconcile_cluster_tags_with_macro("CALL", "PUT", snap)
    assert us is None
    assert eu == "PUT"
    assert changed is True
    assert "MACRO_US_SKIP" in note


def test_reconcile_risk_on_omits_eu_when_quant_flat():
    snap = build_macro_snapshot(
        ["OTC_SPC", "OTC_NDX"],
        ["OTC_FCHI", "OTC_GDAXI"],
        {
            "OTC_SPC": [100.0, 105.0],
            "OTC_NDX": [100.0, 104.0],
            "OTC_FCHI": [100.0, 100.05],
            "OTC_GDAXI": [100.0, 100.04],
        },
        {"min_indices_for_vote": 2, "cluster_min_move_pct": 0.10},
    )
    snap = type(snap)(
        us_dir="up",
        eu_dir="flat",
        us_strength=1.0,
        eu_strength=0.66,
        tag="risk_on",
        eurusd_bias=snap.eurusd_bias,
        cluster_status=snap.cluster_status,
        macro_block=snap.macro_block,
        fx_reference_line=snap.fx_reference_line,
        us_parts=snap.us_parts,
        eu_parts=snap.eu_parts,
    )
    us, eu, changed, note = reconcile_cluster_tags_with_macro("CALL", "PUT", snap)
    assert us == "CALL"
    assert eu is None
    assert changed is True
    assert "MACRO_EU_SKIP" in note


def test_reconcile_cluster_tags_risk_on_forces_call_on_both():
    snap = build_macro_snapshot(
        ["OTC_SPC", "OTC_NDX"],
        ["OTC_FCHI", "OTC_GDAXI"],
        {
            "OTC_SPC": [100.0, 105.0],
            "OTC_NDX": [100.0, 104.0],
            "OTC_FCHI": [100.0, 106.0],
            "OTC_GDAXI": [100.0, 105.5],
        },
        {"min_indices_for_vote": 2, "confluence_conviction_floor": 0.55},
    )
    assert snap.tag == "risk_on"
    us, eu, changed, note = reconcile_cluster_tags_with_macro("PUT", "CALL", snap)
    assert changed is True
    assert us == "CALL"
    assert eu == "CALL"
    assert "MACRO_CLUSTER_ALIGN" in note


def test_reconcile_cluster_tags_divergence_aligns_per_region():
    snap = build_macro_snapshot(
        ["OTC_SPC"],
        ["OTC_FCHI"],
        {"OTC_SPC": [100.0, 105.0], "OTC_FCHI": [100.0, 95.0]},
        {"min_indices_for_vote": 1},
    )
    assert snap.tag == "divergence_us_leads"
    us, eu, changed, _ = reconcile_cluster_tags_with_macro("PUT", "CALL", snap)
    assert changed is True
    assert us == "CALL"
    assert eu == "PUT"


def test_reconcile_cluster_tags_disabled_and_risk_off():
    snap = build_macro_snapshot(
        ["OTC_SPC"],
        ["OTC_FCHI"],
        {"OTC_SPC": [100.0, 95.0], "OTC_FCHI": [100.0, 94.0]},
        {"min_indices_for_vote": 1},
    )
    assert snap.tag == "risk_off"
    us, eu, changed, note = reconcile_cluster_tags_with_macro(
        "CALL",
        "CALL",
        snap,
        {"align_clusters_with_macro_vote": False},
    )
    assert changed is False
    assert note == ""
    assert us == "CALL"
    us2, eu2, changed2, _ = reconcile_cluster_tags_with_macro("CALL", "CALL", snap)
    assert changed2 is True
    assert us2 == "PUT"
    assert eu2 == "PUT"
