from src.domain.risk.dlambert_sizing import dlambert_log_suffix


def test_dlambert_log_suffix_soft_recovery_and_empty():
    suffix = dlambert_log_suffix(
        "D'ALEMBERT",
        108.62,
        93.19,
        10.0,
        consecutive_losses_linear=2,
        payout=0.95,
    )
    assert "soft=2.05x^2" in suffix
    assert "p=0.95" in suffix
    assert "U=$10.00" in suffix
    fixed = dlambert_log_suffix(
        "D'ALEMBERT",
        17.25,
        6.75,
        15.0,
        consecutive_losses_linear=3,
        payout=0.95,
    )
    assert "fixed=U+15%" in fixed
    assert "n=3" in fixed
    assert dlambert_log_suffix("KELLY", 55.0, 0.0, 55.0) == ""
