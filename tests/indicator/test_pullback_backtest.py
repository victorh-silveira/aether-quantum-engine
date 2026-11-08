"""Relatorio diario do backtest por proxy de velas."""

from datetime import UTC, datetime

from indicator.pullback_backtest import format_daily_backtest


def test_backtest_distingue_hoje_sem_dados_e_ultima_sessao():
    now = int(datetime(2026, 10, 26, 20, tzinfo=UTC).timestamp())
    report = {
        "OTC_AEX:900": {
            "estado": "PROXY DE VELAS",
            "ultima_vela_brt": "2026-10-26",
            "setups": 1,
            "acertos": 1,
            "erros": 0,
            "empates": 0,
            "por_dia_brt": {"2026-10-26": {"setups": 1, "acertos": 1, "erros": 0, "empates": 0}},
        },
        "OTC_SSMI:900": {
            "estado": "PROXY DE VELAS",
            "ultima_vela_brt": "2026-10-23",
            "setups": 0,
            "acertos": 0,
            "erros": 0,
            "empates": 0,
            "por_dia_brt": {},
        },
        "OTC_HSI:900": {"estado": "COLETA INDISPONIVEL"},
        "OTC_AEX:3600": {"estado": "PROXY DE VELAS", "setups": 2},
    }
    result = format_daily_backtest(report, now, payout=0.68)
    assert "hoje=2026-10-26 BRT" in result
    assert "OTC_AEX   | 2026-10-26  | 1" in result
    assert "OTC_SSMI  | 2026-10-23  | s/d" in result
    assert "OTC_HSI" in result
    assert "OTC_AEX:3600" not in result
    assert "2026-10-26 | 1" in result
    assert "+0.68" in result
    assert "sem cotacao" in result


def test_backtest_sem_entradas_explicita_amostra_vazia():
    result = format_daily_backtest({}, 0, 0.68)
    assert "Nenhum setup M15 no historico consultado." in result
