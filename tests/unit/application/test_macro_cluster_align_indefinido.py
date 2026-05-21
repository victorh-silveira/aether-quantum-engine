from src.application.services.llm.global_macro_confluence import (
    build_macro_snapshot,
    empty_macro_snapshot,
    reconcile_cluster_tags_with_macro,
)


def test_reconcile_indefinido_aligns_eu_to_quant_call():
    snap = build_macro_snapshot(
        ["OTC_SPC"],
        ["OTC_FCHI"],
        {"OTC_SPC": [100.0, 100.05], "OTC_FCHI": [100.0, 105.0]},
        {"min_indices_for_vote": 1, "cluster_min_move_pct": 0.10},
    )
    snap = type(snap)(
        us_dir="flat",
        eu_dir="up",
        us_strength=0.0,
        eu_strength=1.0,
        tag="indefinido",
        eurusd_bias=snap.eurusd_bias,
        cluster_status=snap.cluster_status,
        macro_block=snap.macro_block,
        fx_reference_line=snap.fx_reference_line,
        us_parts=snap.us_parts,
        eu_parts=snap.eu_parts,
    )
    us, eu, changed, note = reconcile_cluster_tags_with_macro("CALL", "PUT", snap)
    assert us is None
    assert eu == "CALL"
    assert changed is True
    assert "MACRO_US_SKIP" in note


def test_reconcile_indefinido_blocks_weaker_opposing_region():
    snap = empty_macro_snapshot()
    snap = type(snap)(
        us_dir="down",
        eu_dir="up",
        us_strength=0.8,
        eu_strength=0.6,
        tag="indefinido",
        eurusd_bias="PUT",
        cluster_status="",
        macro_block="",
        fx_reference_line=snap.fx_reference_line,
        us_parts=(),
        eu_parts=(),
    )
    us, eu, changed, note = reconcile_cluster_tags_with_macro("CALL", "CALL", snap)
    assert us == "PUT"
    assert eu is None
    assert changed is True
    assert "MACRO_INDEF block_eu_opposing" in note

    snap2 = type(snap)(
        us_dir="down",
        eu_dir="up",
        us_strength=0.4,
        eu_strength=0.9,
        tag="indefinido",
        eurusd_bias="PUT",
        cluster_status="",
        macro_block="",
        fx_reference_line=snap.fx_reference_line,
        us_parts=(),
        eu_parts=(),
    )
    us2, eu2, changed2, note2 = reconcile_cluster_tags_with_macro("CALL", "CALL", snap2)
    assert us2 is None
    assert eu2 == "CALL"
    assert changed2 is True
    assert "MACRO_INDEF block_us_opposing" in note2


def test_reconcile_indefinido_cluster_align_fallback_note():
    snap = build_macro_snapshot(
        ["OTC_SPC"],
        ["OTC_FCHI"],
        {"OTC_SPC": [100.0, 105.0], "OTC_FCHI": [100.0, 106.0]},
        {"min_indices_for_vote": 1},
    )
    snap = type(snap)(
        us_dir="up",
        eu_dir="up",
        us_strength=1.0,
        eu_strength=1.0,
        tag="indefinido",
        eurusd_bias=snap.eurusd_bias,
        cluster_status=snap.cluster_status,
        macro_block=snap.macro_block,
        fx_reference_line=snap.fx_reference_line,
        us_parts=snap.us_parts,
        eu_parts=snap.eu_parts,
    )
    us, eu, changed, note = reconcile_cluster_tags_with_macro("PUT", "PUT", snap)
    assert us == "CALL"
    assert eu == "CALL"
    assert changed is True
    assert "MACRO_CLUSTER_ALIGN" in note


def test_reconcile_indefinido_forces_us_put_when_quant_down_eu_flat():
    snap = build_macro_snapshot(
        ["OTC_SPC", "OTC_NDX"],
        ["OTC_FCHI"],
        {
            "OTC_SPC": [100.0, 95.0],
            "OTC_NDX": [100.0, 94.0],
            "OTC_FCHI": [100.0, 100.05],
        },
        {"min_indices_for_vote": 2, "cluster_min_move_pct": 0.10},
    )
    snap = type(snap)(
        us_dir="down",
        eu_dir="flat",
        us_strength=1.0,
        eu_strength=0.0,
        tag="indefinido",
        eurusd_bias=snap.eurusd_bias,
        cluster_status=snap.cluster_status,
        macro_block=snap.macro_block,
        fx_reference_line=snap.fx_reference_line,
        us_parts=snap.us_parts,
        eu_parts=snap.eu_parts,
    )
    us, eu, changed, _ = reconcile_cluster_tags_with_macro("CALL", "PUT", snap)
    assert changed is True
    assert us == "PUT"
    assert eu is None


def test_reconcile_skips_us_when_quant_flat():
    snap = build_macro_snapshot(
        ["OTC_SPC"],
        ["OTC_FCHI"],
        {"OTC_SPC": [100.0, 100.0], "OTC_FCHI": [100.0, 105.0]},
        {"min_indices_for_vote": 1},
    )
    snap = type(snap)(
        us_dir="flat",
        eu_dir="up",
        us_strength=0.0,
        eu_strength=1.0,
        tag="indefinido",
        eurusd_bias=snap.eurusd_bias,
        cluster_status=snap.cluster_status,
        macro_block=snap.macro_block,
        fx_reference_line=snap.fx_reference_line,
        us_parts=snap.us_parts,
        eu_parts=snap.eu_parts,
    )
    us, eu, changed, note = reconcile_cluster_tags_with_macro("PUT", "CALL", snap)
    assert changed is True
    assert us is None
    assert eu == "CALL"
    assert "MACRO_US_SKIP" in note
