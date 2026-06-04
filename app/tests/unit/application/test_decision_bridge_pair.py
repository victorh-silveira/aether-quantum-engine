from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from src.application.services.deep_learning.decision_bridge import _pair_prices_for_symbol


def test_pair_prices_range_hedge_peers():
    orch = SimpleNamespace(symbols=["R_10", "R_100"])
    orch.stream = MagicMock()
    orch.stream.get_numpy_series = MagicMock(
        side_effect=lambda sym, _col: np.array([1.0, 2.0]) if sym == "R_100" else np.array([3.0, 4.0])
    )
    peer = _pair_prices_for_symbol(orch, "R_10")
    assert peer is not None and len(peer) == 2


def test_pair_prices_none_for_single_symbol():
    orch = SimpleNamespace(symbols=["R_50"])
    orch.stream = MagicMock()
    assert _pair_prices_for_symbol(orch, "R_50") is None
