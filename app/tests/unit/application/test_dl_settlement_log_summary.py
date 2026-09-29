"""A telemetria de treino nao inventa settlement quando n=0."""

from src.application.services.deep_learning.dl_symbol_train_success import _settlement_log_summary


def test_missing_broker_settlement_is_reported_as_unavailable():
    summary = _settlement_log_summary(
        {
            "deploy_settlement_n": 0,
            "deploy_settlement_source": "broker_audit_required",
            "deploy_settlement_win_rate": 0.0,
        }
    )
    assert "settle_n=0" in summary
    assert "source=broker_audit_required" in summary
    assert "settle_wr=NA" in summary
    assert "settle_lcb90=NA" in summary
    assert "settle_brier=NA" in summary
    assert "label_wr=NA" in summary


def test_measured_proxy_is_explicitly_labeled_with_source_and_sample_count():
    summary = _settlement_log_summary(
        {
            "deploy_settlement_n": 171,
            "deploy_settlement_source": "m5_close_proxy",
            "deploy_settlement_win_rate": 0.5848,
            "deploy_settlement_wilson_lcb": 0.522,
            "deploy_settlement_brier": 0.245,
            "deploy_label_win_rate": 0.5848,
        }
    )
    assert "settle_n=171 source=m5_close_proxy" in summary
    assert "settle_wr=0.58" in summary
    assert "settle_lcb90=0.52" in summary
    assert "settle_brier=0.245" in summary
    assert "label_wr=0.58" in summary
