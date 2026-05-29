from scripts.backtest.backtest_cluster_runtime import BacktestClusterRuntime
from src.domain.models.trade import TradeDirection


def test_runtime_pause_and_quarantine_after_loss():
    cfg = {"orchestrator": {"cluster_pause_after_loss_cycles": 2}}
    rt = BacktestClusterRuntime(cfg, symbols=["OTC_SPC"], anchor="frxEURUSD")
    rt.begin_cycle()
    assert rt._cluster_pause_after_loss_active is False
    rt.on_trade_loss(symbol="OTC_SPC", direction=TradeDirection.CALL)
    rt.end_cycle()
    rt.begin_cycle()
    assert rt._cluster_pause_after_loss_active is True
    rt.end_cycle()
    rt.begin_cycle()
    assert rt._cluster_pause_after_loss_active is True
    rt.end_cycle()
    rt.begin_cycle()
    assert rt._cluster_pause_after_loss_active is False
