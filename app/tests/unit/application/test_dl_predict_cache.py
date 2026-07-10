from types import SimpleNamespace

import numpy as np
import pytest

from src.application.services.deep_learning.dl_model_types import FeatureNormStats
from src.application.services.deep_learning.dl_predict_cache import (
    prediction_cache,
    resolve_cached_prediction,
    store_prediction_cache,
)
from src.infrastructure.inference.triton_tensor_builder import (
    PartialInferenceHistoryError,
    build_inference_tensor,
    inference_tensor_fingerprint,
    resolve_sequence_end_index,
)


def test_resolve_sequence_end_index_raises_on_short_history():
    with pytest.raises(PartialInferenceHistoryError):
        resolve_sequence_end_index(4, 8)


def test_inference_tensor_fingerprint_is_stable():
    tensor = np.ones((1, 8, 4), dtype=np.float32)
    assert inference_tensor_fingerprint(tensor) == inference_tensor_fingerprint(tensor.copy())


def test_store_and_resolve_cached_prediction_outside_boundary():
    orch = SimpleNamespace(_dl_prediction_cache={})
    entry = {"direction": "CALL", "metrics": {"raw_prob": 0.62}}
    fingerprint = b"tensor-fingerprint"
    store_prediction_cache(orch, "RDBULL", entry, tensor_fingerprint=fingerprint, boundary_epoch=1000)
    cached = resolve_cached_prediction(orch, "RDBULL", at_boundary=False)
    assert cached is entry


def test_resolve_cached_prediction_reuses_duplicate_tensor_on_boundary():
    orch = SimpleNamespace(_dl_prediction_cache={})
    entry = {"direction": "PUT", "metrics": {"raw_prob": 0.41}}
    fingerprint = b"duplicate-tensor"
    store_prediction_cache(orch, "RDBEAR", entry, tensor_fingerprint=fingerprint, boundary_epoch=1000)
    cached = resolve_cached_prediction(
        orch,
        "RDBEAR",
        at_boundary=True,
        tensor_fingerprint=fingerprint,
    )
    assert cached is entry


def test_resolve_cached_prediction_rejects_mismatch_on_boundary():
    orch = SimpleNamespace(_dl_prediction_cache={})
    entry = {"direction": "CALL"}
    store_prediction_cache(orch, "RDBULL", entry, tensor_fingerprint=b"old", boundary_epoch=1)
    assert (
        resolve_cached_prediction(
            orch,
            "RDBULL",
            at_boundary=True,
            tensor_fingerprint=b"new",
        )
        is None
    )


def test_resolve_cached_prediction_requires_fingerprint_on_boundary():
    orch = SimpleNamespace(_dl_prediction_cache={})
    store_prediction_cache(orch, "RDBULL", {"direction": "CALL"}, tensor_fingerprint=b"x", boundary_epoch=1)
    assert resolve_cached_prediction(orch, "RDBULL", at_boundary=True, tensor_fingerprint=None) is None


def test_resolve_cached_prediction_ignores_invalid_entry():
    orch = SimpleNamespace(_dl_prediction_cache={"RDBULL": {"entry": "invalid"}})
    assert resolve_cached_prediction(orch, "RDBULL", at_boundary=False) is None


def test_prediction_cache_initializes_on_orchestrator():
    orch = SimpleNamespace()
    cache = prediction_cache(orch)
    assert cache is orch._dl_prediction_cache
    assert cache == prediction_cache(orch)


def test_build_inference_tensor_raises_partial_history():
    stats = FeatureNormStats(
        mean=np.zeros(34, dtype=np.float32),
        std=np.ones(34, dtype=np.float32),
    )
    with pytest.raises(PartialInferenceHistoryError):
        build_inference_tensor(np.array([1.0, 2.0]), 8, stats)
