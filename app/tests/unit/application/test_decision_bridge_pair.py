from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from src.application.services.deep_learning.decision_bridge import _pair_prices_for_symbol


def test_pair_prices_rdbull_rdbear():
    orch = SimpleNamespace(symbols=["RDBULL", "RDBEAR"])
    orch.stream = MagicMock()
    orch.stream.get_numpy_series = MagicMock(
        side_effect=lambda sym, _col: np.array([1.0, 2.0]) if sym == "RDBEAR" else np.array([3.0, 4.0])
    )
    peer = _pair_prices_for_symbol(orch, "RDBULL")
    assert peer is not None and len(peer) == 2


def test_pair_prices_none_for_single_symbol():
    orch = SimpleNamespace(symbols=["RDBULL"])
    orch.stream = MagicMock()
    assert _pair_prices_for_symbol(orch, "RDBULL") is None
