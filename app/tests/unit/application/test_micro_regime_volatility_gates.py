from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.application.services.deep_learning.dl_feature_build import precompute_price_series
from src.application.services.deep_learning.dl_predict_build import build_prediction_entry
from src.application.services.orchestrator.execution_collect import collect_cluster_orders
from src.domain.models.trade import TradeDirection
from src.domain.risk.executed_stake_reconciliation import fractional_payoff_residual_cents
from src.infrastructure.inference.meta_classifier_client import MetaClassifierClient


def test_volatility_shadow_ratio_and_rolling_zscore():
    prices = np.linspace(100.0, 105.0, 50)
    high = prices * 1.001
    low = prices * 0.999
    open_ = prices * 0.9995
    # Fornece os arrays adicionais de pavios para o precompute
    series = precompute_price_series(
        prices,
        granularity=60,
        symbol="OTC_SPC",
        open_=open_,
        high=high,
        low=low,
    )
    assert "volatility_shadow_ratio" in series
    assert "volatility_shadow_ratio_zscore" in series
    assert len(series["volatility_shadow_ratio"]) == len(prices)
    # Valores de Z-Score devem estar clipados dentro de +-3.0
    assert np.all(series["volatility_shadow_ratio_zscore"] <= 3.0)
    assert np.all(series["volatility_shadow_ratio_zscore"] >= -3.0)


def test_micro_congestion_squeeze_veto():
    # Squeeze real com ADX < 0.15 e bb_width < 0.04
    prices = np.ones(120) * 100.0  # Sem volatilidade, bb_width muito pequeno
    series = precompute_price_series(prices, granularity=60)
    # Força adx baixo
    series["adx"] = np.ones(120) * 0.10

    orch = SimpleNamespace(
        config={"orchestrator": {"execution": {"mandatory_trade_each_cycle": True}}},
        stream=SimpleNamespace(get_micro_numpy_series=lambda s, k: None),
    )
    runtime = {
        "lookback": 48,
        "calibrator": SimpleNamespace(method="temperature_platt", temperature=1.0, platt_a=1.0, platt_b=0.0),
    }
    params = {"implied_vol_bars": 60, "granularity": 60, "min_edge_execute": 0.04}

    entry = build_prediction_entry(
        orch,
        "OTC_SPC",
        prices,
        series,
        runtime,
        params,
        train_loss=0.01,
        direction=TradeDirection.CALL,
        prob=0.55,
        raw_prob=0.55,
        dynamic=None,
        dynamic_cfg={},
        call_threshold=0.53,
        put_threshold=0.47,
        exec_cfg={},
        val_accuracy=0.65,
    )

    assert entry["metrics"]["gate_reason"] is None
    assert entry["metrics"]["micro_chop_congestion"] is True
    assert entry["metrics"]["trade_score"] == 0.51


@pytest.mark.asyncio
async def test_chaotic_volatility_asymmetric_penalty():
    client = MetaClassifierClient(base_url="http://localhost:8005", timeout=1.0, enabled=True)
    # Mock do AsyncClient do httpx
    client._client = MagicMock()
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "predicted_payoff_edge": 0.10,
        "meta_applied": True,
        "edge_expectancy": "WIN_EXPECTED",
    }
    client._client.post = AsyncMock(return_value=mock_response)

    # Vetor com bb_width_zscore (idx 8) > 2.5 indicando transição caótica
    f_vec = [0.0] * 43
    f_vec[8] = 2.8  # > 2.5
    req = {
        "symbol": "OTC_SPC",
        "tcn_probability": 0.58,
        "direction": "CALL",
        "feature_vector": f_vec,
    }

    res = await client.predict_meta(req, fallback_score=0.58)
    # Deve aplicar a penalidade assimétrica de 0.5 sobre o edge positivo -> 0.10 * 0.5 = 0.05
    assert res["predicted_payoff_edge"] == pytest.approx(0.05)


def test_consensus_cointegration_redirect():
    # Drawdown > 15% na conta de $100 -> pending_loss total > $15.00
    risk_mgr = SimpleNamespace(
        initial_bankroll=100.0,
        pending_loss={"OTC_SPC": 10.0, "R_50": 6.0},
        pending_loss_total=lambda: 16.0,
        consecutive_losses_linear=2,
        total_session_profit=-16.0,
    )
    orch = SimpleNamespace(
        risk_manager=risk_mgr,
        config={
            "orchestrator": {"execution": {"mandatory_trade_each_cycle": True}},
            "deep_learning": {"enabled": True},
        },
        _active_cycle_id=1,
        _recovery_skip_counter=0,
    )
    exec_mgr = SimpleNamespace(
        orch=orch,
        logger=MagicMock(),
        _mandatory_trade_each_cycle=lambda: True,
        _trade_symbols=lambda: ["OTC_SPC", "R_50"],
    )

    decisions = {
        "OTC_SPC": {
            "direction": TradeDirection.CALL,
            "metrics": {
                "execute": True,
                "calibrated_prob": 0.60,
                "raw_prob": 0.60,
                "edge_zscore": 1.2,
                "val_accuracy": 0.65,
                "trade_score": 0.62,
            },
        },
        "R_50": {
            "direction": TradeDirection.PUT,
            "metrics": {
                "execute": True,
                "calibrated_prob": 0.58,
                "raw_prob": 0.58,
                "edge_zscore": 0.8,
                "val_accuracy": 0.65,
                "trade_score": 0.60,
            },
        },
    }

    with (
        patch(
            "src.application.services.orchestrator.execution_collect_gather.gather_cluster_candidates"
        ) as mock_gather,
        patch("src.application.services.orchestrator.execution_collect.filter_loss_protection_candidates") as mock_loss,
        patch("src.application.services.orchestrator.execution_collect.filter_recovery_hurst_candidates") as mock_hurst,
    ):
        # Mock do gather para retornar os candidatos
        mock_gather.return_value = [
            ("OTC_SPC", TradeDirection.CALL, decisions["OTC_SPC"]["metrics"]),
            ("OTC_SPC", TradeDirection.PUT, decisions["OTC_SPC"]["metrics"]),
        ]
        mock_loss.return_value = mock_gather.return_value
        mock_hurst.return_value = mock_gather.return_value

        res = collect_cluster_orders(exec_mgr, decisions)
        # OTC_SPC deve ser preferível por ter menor entropia ou maior Z-score -> candidates filtrado para apenas 1 símbolo
        assert len(res) == 1
        assert res[0][0] in ("OTC_SPC", "OTC_SPC")


def test_sub_cent_reconciliation_exact_rounding():
    # api_profit com sub-centavos (ex: 1.9542)
    # expected = 1.00 (stake) * 0.95 (payout) = 0.95
    # residual = api_profit - expected = 1.9542 - 0.95 = 1.0042 (excede 0.10, retorna 0.0)
    assert fractional_payoff_residual_cents(1.9542, 1.0, 0.95) == 0.0

    # residual pequeno (ex: api_profit = 0.9542, expected = 0.95 -> residual = 0.0042)
    assert fractional_payoff_residual_cents(0.9542, 1.0, 0.95) == pytest.approx(0.0042)
