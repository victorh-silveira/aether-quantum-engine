"""Testes unitarios para ranqueamento e coleta do setup pullback."""

import asyncio

import pytest

from indicator.pullback_promotion import format_ranking_table, rank_m5_pairs
from indicator.training import _fetch_candles, history


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
    assert "EV PROXY" in table
    assert "Payout hipotetico do proxy" in table


def test_ranking_sem_amostra_nao_exibe_ev():
    ranked = rank_m5_pairs({"OTC_SSMI:900": {"estado": "PROXY DE VELAS", "setups": 0}}, payout=0.68)
    table = format_ranking_table(ranked)
    assert "SEM AMOSTRA" in table
    assert "-1.0000" not in table
    assert "68.00%" in table


def test_historico_repete_limite_sem_engolir_erro(monkeypatch):
    calls = 0
    delays = []

    class Source:
        async def candles(self, symbol, count, end, period):
            nonlocal calls
            calls += 1
            if calls < 3:
                raise RuntimeError("You have reached the rate limit for ticks_history_expensive.")
            return [{"epoch": 900}]

    async def sleep(delay):
        delays.append(delay)

    monkeypatch.setattr("indicator.training.asyncio.sleep", sleep)
    assert asyncio.run(_fetch_candles(Source(), "OTC_SSMI", 1000, "latest", 900)) == [{"epoch": 900}]
    assert calls == 3
    assert delays == [2, 5]

    class FailingSource:
        async def candles(self, symbol, count, end, period):
            raise RuntimeError("other error")

    with pytest.raises(RuntimeError, match="other error"):
        asyncio.run(_fetch_candles(FailingSource(), "OTC_SSMI", 1000, "latest", 900))


def test_historico_otc_nao_repara_pausa_entre_dias():
    calls = []

    class Source:
        async def candles(self, symbol, count, end, period):
            calls.append((count, end))
            if end == "latest":
                epochs = [0, 900, 86400, 87300, 89100]
            else:
                epochs = [88200]
            return [{"epoch": epoch, "open": 1, "high": 2, "low": 1, "close": 2} for epoch in epochs]

    result = asyncio.run(history(Source(), "OTC_SSMI", 90000, pages=1, period=900))
    assert [row.epoch for row in result] == [0, 900, 86400, 87300, 88200, 89100]
    assert calls == [(1000, "latest"), (3, 89100 - 1)]


def test_historico_limita_reparos_a_tres_lacunas():
    calls = []

    class Source:
        async def candles(self, symbol, count, end, period):
            calls.append(end)
            epochs = [0, 1800, 3600, 5400, 7200] if end == "latest" else []
            return [{"epoch": epoch, "open": 1, "high": 2, "low": 1, "close": 2} for epoch in epochs]

    result = asyncio.run(history(Source(), "R_25", 8100, pages=1, period=900))
    assert len(result) == 5
    assert calls == ["latest", 1799, 3599, 5399]
