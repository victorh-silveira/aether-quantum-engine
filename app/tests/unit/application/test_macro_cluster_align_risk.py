from src.application.services.llm.macro_cluster_align import expected_cluster_tags_line


def test_expected_cluster_tags_line_risk_on():
    line = expected_cluster_tags_line(
        tag="risk_on",
        us_dir="up",
        eu_dir="up",
        us_strength=0.7,
        eu_strength=0.65,
        macro_cfg={"confluence_conviction_floor": 0.55},
    )
    assert "CLUSTER_QUANT_REF" in line
    assert "CALL" in line


def test_expected_cluster_tags_line_risk_off():
    line = expected_cluster_tags_line(
        tag="risk_off",
        us_dir="down",
        eu_dir="down",
        us_strength=0.8,
        eu_strength=0.75,
        macro_cfg={"confluence_conviction_floor": 0.55},
    )
    assert "PUT" in line
