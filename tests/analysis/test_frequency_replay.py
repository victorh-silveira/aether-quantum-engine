"""Contratos do replay de cluster sem credenciais nem contratos reais."""

import math
from datetime import UTC, datetime

import pytest

from scripts.analysis.frequency_replay import Signal, replay_portfolio
from src.domain.fundamental.economic_calendar import CalendarPolicy, EconomicEvent
from src.domain.risk.pullback_stake import pullback_stake


EPOCH = int(datetime(2026, 4, 1, 12, tzinfo=UTC).timestamp())


def test_cluster_reserves_before_next_order_and_settles_simultaneously():
    """Quatro compras usam disponibilidades decrescentes, sem ganhos intracluster."""
    signals = [Signal(symbol, EPOCH, "CALL", 1) for symbol in ("D", "C", "B", "OTC_FTSE", "A")]
    result = replay_portfolio(signals, payout=0.7)
    assert [trade["symbol"] for trade in result["trades"]] == ["OTC_FTSE", "A", "B", "C"]
    reserved = 0.0
    expected = []
    for _ in range(4):
        stake, _ = pullback_stake(
            1000 - reserved,
            0,
            0.7,
            1,
            amort_steps=0,
            base_stake_pct=0.015,
            cycle_risk_budget_pct=0.04,
            cycle_order_count=4,
        )
        expected.append(stake)
        reserved += stake
    assert [trade["stake"] for trade in result["trades"]] == expected
    assert reserved <= 40
    assert result["cycles"][0]["snapshot"] == 1000
    assert result["counts"]["capacity_rejections"] == 1
    assert result["final_bankroll"] == round(1000 + sum(round(stake * 0.7, 2) for stake in expected), 2)


def test_loss_tie_and_next_cycle_balance():
    """Empate devolve stake e o proximo ciclo usa patrimonio liquidado."""
    signals = [Signal("A", EPOCH, "PUT", -1), Signal("B", EPOCH, "CALL", 0), Signal("A", EPOCH + 900, "PUT", -1)]
    result = replay_portfolio(signals, payout=0.7235)
    assert result["cycles"][1]["snapshot"] == result["cycles"][0]["balance"]
    assert result["trades"][1]["pnl"] == 0
    assert result["max_consecutive_losses"] == 1
    assert result["max_drawdown_pct"] > 0
    assert result["net_units"] == -2
    assert result["qualifies_contract"] is False


def test_calendar_gate_and_no_artificial_missing_coverage_veto():
    """Cobertura ausente e declarada, sem inventar bloqueios macro."""
    event = EconomicEvent("Release", "USD", EPOCH + 900, "HIGH")
    signals = [Signal("OTC_FTSE", EPOCH, "CALL", 1)]
    result = replay_portfolio(signals, payout=0.7, events=(event,))
    assert result["trade_count"] == 0
    assert result["counts"]["calendar_vetoes"] == 1
    assert not result["calendar_coverage_complete"]
    assert replay_portfolio(signals, payout=0.7)["trade_count"] == 1
    assert (
        replay_portfolio(signals, payout=0.7, events=(event,), calendar_policy=CalendarPolicy(enabled=False))[
            "trade_count"
        ]
        == 1
    )


def test_provided_score_precedes_anchor_and_outcome_never_affects_selection():
    """Score historico informado precede fallback e empate independe do futuro."""
    signals = [Signal("OTC_FTSE", EPOCH, "CALL", 1, (0.0,)), Signal("A", EPOCH, "CALL", -1, (1.0,))]
    result = replay_portfolio(signals, payout=0.8)
    assert result["ranking"] == "provided_score"
    assert result["trades"][0]["symbol"] == "A"
    flipped = [Signal(s.symbol, s.entry_epoch, s.side, -s.outcome, s.selection_score) for s in signals]
    assert [t["symbol"] for t in replay_portfolio(flipped, payout=0.8)["trades"]] == [
        t["symbol"] for t in result["trades"]
    ]


def test_daily_window_includes_zero_days_and_filters_entries():
    """Media diaria inclui datas sem entradas e respeita janela BRT."""
    result = replay_portfolio(
        [Signal("A", EPOCH, "CALL", 1), Signal("B", EPOCH + 3 * 86400, "CALL", 1)],
        payout=0.7,
        start_epoch=EPOCH,
        end_epoch=EPOCH + 2 * 86400,
    )
    assert len(result["daily"]) == 3
    assert result["zero_trade_days"] == 2
    assert result["mean_daily_trades"] == pytest.approx(1 / 3)
    assert result["active_days"] == 1


def test_no_signals_and_tiny_bankroll():
    """Banca abaixo do minimo bloqueia compra sem saldo negativo."""
    assert replay_portfolio([], payout=0.7)["trade_count"] == 0
    result = replay_portfolio([Signal("A", EPOCH, "CALL", -1)], payout=0.7, initial_bankroll=1)
    assert result["trade_count"] == 0
    assert result["counts"]["stake_rejections"] == 1
    assert result["final_bankroll"] == 1


@pytest.mark.parametrize(
    "kwargs",
    [{"side": "X"}, {"outcome": 3}, {"symbol": ""}, {"entry_epoch": EPOCH + 1}, {"selection_score": (math.nan,)}],
)
def test_invalid_signal(kwargs):
    """Dados ambiguos sao rejeitados antes da selecao."""
    values = {"symbol": "A", "entry_epoch": EPOCH, "side": "CALL", "outcome": 1}
    with pytest.raises(ValueError):
        Signal(**(values | kwargs))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"payout": 0},
        {"payout": math.inf},
        {"initial_bankroll": 0},
        {"initial_bankroll": math.nan},
        {"start_epoch": EPOCH + 900, "end_epoch": EPOCH},
    ],
)
def test_invalid_replay_parameters(kwargs):
    """Payout e banca invalidos jamais resultam em compras."""
    with pytest.raises(ValueError):
        replay_portfolio([], **({"payout": 0.7} | kwargs))


def test_duplicate_cycle_symbol_is_rejected():
    """Um ativo nao pode gerar dois contratos no mesmo ciclo."""
    signal = Signal("A", EPOCH, "CALL", 1)
    with pytest.raises(ValueError, match="duplicado"):
        replay_portfolio([signal, signal], payout=0.7)


def test_small_bankroll_exposes_production_minimum_budget_conflict():
    """Auditoria nao oculta o piso absoluto que ultrapassa budget em banca pequena."""
    result = replay_portfolio(
        [Signal(symbol, EPOCH, "CALL", 1) for symbol in ("A", "B", "C", "D")],
        payout=0.7,
        initial_bankroll=50,
    )
    assert result["cycles"][0]["reserved_stake"] == 4
    assert result["counts"]["cluster_budget_breaches"] == 1


def test_lexical_ties_with_prefix_symbols():
    """Prefixos preservam ordenacao lexical, independentemente da ordem recebida."""
    result = replay_portfolio([Signal(s, EPOCH, "CALL", 1) for s in ("AA", "A")], payout=0.7)
    assert [trade["symbol"] for trade in result["trades"]] == ["A", "AA"]
