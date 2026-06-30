import numpy as np
import pytest

from src.infrastructure.inference.triton_inference_client import _parse_raw_output


class _FakeResult:
    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def as_numpy(self, _name: str):
        return self._arr


def test_parse_raw_output_clamps_probability():
    assert _parse_raw_output(_FakeResult(np.array([1.5], dtype=np.float32))) == 1.0
    assert _parse_raw_output(_FakeResult(np.array([0.42], dtype=np.float32))) == pytest.approx(0.42)
    assert _parse_raw_output(_FakeResult(None)) == 0.5
