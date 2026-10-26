"""Testes unitarios para ranqueamento e promocao do setup pullback M5."""

from indicator.pullback_promotion import format_ranking_table, rank_m5_pairs


def test_rank_m5_pairs_orders_by_ev_and_win_rate():
    report = {
        "R_25:900": {
            "estado": "PROXY DE VELAS",
            "setups": 20,
            "acertos": 12,
            "erros": 8,
            "empates": 0,
            "max_losses_consecutivos": 4,
        },
        "1HZ25V:900": {
            "estado": "PROXY DE VELAS",
            "setups": 30,
            "acertos": 21,
            "erros": 9,
            "empates": 0,
            "max_losses_consecutivos": 3,
        },
        "R_50:900": {
            "estado": "SEM SINAL",
            "setups": 0,
        },
        "R_25:3600": {
            "estado": "PROXY DE VELAS",
            "setups": 15,
            "acertos": 10,
            "erros": 5,
            "max_losses_consecutivos": 2,
        },
        "1HZ10V:900": {
            "estado": "PROXY DE VELAS",
            "setups": 0,
            "acertos": 0,
            "erros": 0,
        },
    }
    excluded = {"R_100"}
    ranked = rank_m5_pairs(report, excluded_symbols=excluded, payout=0.85)
    assert len(ranked) == 3
    assert ranked[0]["symbol"] == "1HZ25V"
    assert ranked[0]["eligible"] is True
    assert ranked[0]["win_rate"] == 0.70
    assert ranked[0]["max_consecutive_losses"] == 3
    assert ranked[0]["ev"] > ranked[1]["ev"]

    excluded_ranked = rank_m5_pairs(report, excluded_symbols={"1HZ25V"}, payout=0.85)
    assert excluded_ranked[0]["symbol"] == "R_25"


def test_format_ranking_table():
    ranked = [
        {
            "symbol": "1HZ25V",
            "period": 900,
            "setups": 20,
            "wins": 14,
            "losses": 6,
            "ties": 0,
            "max_consecutive_losses": 2,
            "win_rate": 0.70,
            "ev": 0.295,
            "eligible": True,
            "excluded": False,
        },
        {
            "symbol": "R_50",
            "period": 900,
            "setups": 10,
            "wins": 4,
            "losses": 6,
            "ties": 0,
            "max_consecutive_losses": 4,
            "win_rate": 0.40,
            "ev": -0.26,
            "eligible": False,
            "excluded": True,
        },
    ]
    table = format_ranking_table(ranked)
    assert "RANKING DO SETUP PULLBACK M15" in table
    assert "1HZ25V" in table
    assert "MAX LOSS" in table
    assert "LIDER PROXY" in table
    assert "EXCLUIDO" in table
