"""Testes de cobertura para gaps em narrative_context."""

from datetime import UTC, datetime
from unittest.mock import patch

import numpy as np

from src.application.services.llm import IndicatorConfig
from src.application.services.llm.narrative_context import (
    _alignment_label_quant,
    _micro_swing_tag,
    _statistical_anomaly_hint,
    describe_session_context,
)


def test_statistical_anomaly_hint_branches():
    """Cobre as ramificações de dica de anomalia estatística."""
    assert "n/d" in _statistical_anomaly_hint(None)
    assert "anomalia_positiva" in _statistical_anomaly_hint(3.0)
    assert "anomalia_negativa" in _statistical_anomaly_hint(-3.0)


def test_alignment_label_quant_reversion():
    """Cobre o retorno de reversao_esperada."""
    cfg = IndicatorConfig(hurst_window=15)
    closes = np.array([100.0, 101.0, 100.0, 101.0, 100.0] * 10)
    assert _alignment_label_quant(closes, cfg) == "Mean Reversion Alpha"


def test_describe_session_context_windows():
    """Cobre janelas de sessão asia, europa, ny e pos_ny."""
    with patch("src.application.services.llm.narrative_context.datetime") as mock_date:
        mock_date.now.return_value = datetime(2024, 1, 1, 3, 0, tzinfo=UTC)
        assert "asia" in describe_session_context()

    with patch("src.application.services.llm.narrative_context.datetime") as mock_date:
        mock_date.now.return_value = datetime(2024, 1, 1, 10, 0, tzinfo=UTC)
        assert "europa" in describe_session_context()

    with patch("src.application.services.llm.narrative_context.datetime") as mock_date:
        mock_date.now.return_value = datetime(2024, 1, 1, 15, 0, tzinfo=UTC)
        assert "ny" in describe_session_context()

    with patch("src.application.services.llm.narrative_context.datetime") as mock_date:
        mock_date.now.return_value = datetime(2024, 1, 1, 22, 0, tzinfo=UTC)
        assert "pos_ny" in describe_session_context()


def test_micro_swing_tag_branches():
    """Cobre as ramificações de tag de micro-swing."""
    assert _micro_swing_tag(np.array([100.0] * 3)) == "n/d"

    assert _micro_swing_tag(np.array([100.0, 95.0, 98.0, 97.0, 99.0, 98.0])) == "pressao_baixa"

    assert _micro_swing_tag(np.array([100.0, 110.0, 90.0, 95.0])) == "misturado"
