"""Regressoes de unidade de volatilidade e payout cotado."""

import pytest

from src.application.services.deep_learning.dl_predict_metrics import indicators_from_series
from src.application.services.market_audit_log_helpers import resolve_raw_predicted_edge


def test_snapshot_converte_atr_relativo_em_preco():
    result = indicators_from_series({"atr_raw": [0.002], "close": [5000.0]})
    assert result["atr_abs"] == pytest.approx(10.0)


def test_edge_raw_respeita_payout_observado():
    metrics = {"raw_prob": 0.55, "payout_assumed": 0.784}
    assert resolve_raw_predicted_edge(metrics, direction="CALL") == pytest.approx(-0.0188)
    assert resolve_raw_predicted_edge(metrics, direction="PUT") == pytest.approx(-0.1972)
