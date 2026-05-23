"""Testes para cobertura de cenários NA em linhas numéricas."""

from src.application.services.llm import IndicatorConfig
from src.application.services.llm.indicators_numeric import _numeric_tf_segment


def test_numeric_tf_segment_na():
    """Valida retorno NA quando há dados insuficientes."""
    cfg = IndicatorConfig()
    res = _numeric_tf_segment("H4", [100.0] * 10, cfg)
    assert "H4=na" in res
