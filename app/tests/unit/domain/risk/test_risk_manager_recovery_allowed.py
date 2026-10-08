from src.domain.risk.risk_manager import RiskManager


def test_recovery_allowed_fails_on_low_val_accuracy(kelly_config):
    kelly_config["dlambert"]["recovery_min_val_accuracy"] = 0.50
    rm = RiskManager(kelly_config)
    rm.pending_loss["R_10"] = 10.0
    dl_metrics = {"deploy_ok": True, "val_accuracy": 0.40, "trade_score": 0.50, "raw_prob": 0.50}
    assert rm._recovery_allowed("R_10", 0.50, dl_metrics=dl_metrics) is False
