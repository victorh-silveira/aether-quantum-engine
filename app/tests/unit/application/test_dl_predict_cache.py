from types import SimpleNamespace

from src.application.services.deep_learning.dl_predict_cache import (
    prediction_cache,
    resolve_cached_prediction,
    store_prediction_cache,
)


def test_store_and_resolve_cached_prediction_outside_boundary():
    orch = SimpleNamespace(_dl_prediction_cache={})
    entry = {"direction": "CALL", "metrics": {"raw_prob": 0.62}}
    fingerprint = b"tensor-fingerprint"
    store_prediction_cache(orch, "R_10", entry, tensor_fingerprint=fingerprint, boundary_epoch=1000)
    cached = resolve_cached_prediction(orch, "R_10", at_boundary=False)
    assert cached is entry
    assert (
        resolve_cached_prediction(
            orch,
            "R_10",
            at_boundary=False,
            tensor_fingerprint=b"stale-tensor",
        )
        is None
    )
    assert (
        resolve_cached_prediction(
            orch,
            "R_10",
            at_boundary=False,
            tensor_fingerprint=fingerprint,
        )
        is entry
    )


def test_resolve_cached_prediction_reuses_duplicate_tensor_on_boundary():
    orch = SimpleNamespace(_dl_prediction_cache={})
    entry = {"direction": "PUT", "metrics": {"raw_prob": 0.41}}
    fingerprint = b"duplicate-tensor"
    store_prediction_cache(orch, "R_10", entry, tensor_fingerprint=fingerprint, boundary_epoch=1000)
    cached = resolve_cached_prediction(
        orch,
        "R_10",
        at_boundary=True,
        tensor_fingerprint=fingerprint,
    )
    assert cached is entry


def test_resolve_cached_prediction_rejects_mismatch_on_boundary():
    orch = SimpleNamespace(_dl_prediction_cache={})
    entry = {"direction": "CALL"}
    store_prediction_cache(orch, "R_10", entry, tensor_fingerprint=b"old", boundary_epoch=1)
    assert (
        resolve_cached_prediction(
            orch,
            "R_10",
            at_boundary=True,
            tensor_fingerprint=b"new",
        )
        is None
    )


def test_resolve_cached_prediction_misses_when_boundary_epoch_advances():
    orch = SimpleNamespace(_dl_prediction_cache={})
    entry = {"direction": "CALL", "metrics": {"raw_prob": 0.62}}
    fingerprint = b"same-tensor"
    store_prediction_cache(orch, "R_10", entry, tensor_fingerprint=fingerprint, boundary_epoch=1000)
    assert (
        resolve_cached_prediction(
            orch,
            "R_10",
            at_boundary=False,
            boundary_epoch=1060,
        )
        is None
    )
    assert (
        resolve_cached_prediction(
            orch,
            "R_10",
            at_boundary=True,
            tensor_fingerprint=fingerprint,
            boundary_epoch=1060,
        )
        is None
    )
    assert (
        resolve_cached_prediction(
            orch,
            "R_10",
            at_boundary=False,
            boundary_epoch=1000,
        )
        is entry
    )
    assert (
        resolve_cached_prediction(
            orch,
            "R_10",
            at_boundary=True,
            tensor_fingerprint=fingerprint,
            boundary_epoch=1000,
        )
        is entry
    )


def test_resolve_cached_prediction_misses_absent_or_bad_epoch():
    orch = SimpleNamespace(_dl_prediction_cache={})
    assert resolve_cached_prediction(orch, "R_10", at_boundary=False, boundary_epoch=1) is None
    store_prediction_cache(
        orch,
        "R_10",
        {"direction": "PUT"},
        tensor_fingerprint=b"fp",
        boundary_epoch=1000,
    )
    orch._dl_prediction_cache["R_10"]["boundary_epoch"] = object()
    assert (
        resolve_cached_prediction(
            orch,
            "R_10",
            at_boundary=False,
            boundary_epoch=1000,
        )
        is None
    )


def test_resolve_cached_prediction_misses_when_cycle_id_changes():
    orch = SimpleNamespace(_dl_prediction_cache={})
    entry = {"direction": "CALL"}
    store_prediction_cache(
        orch,
        "R_10",
        entry,
        tensor_fingerprint=b"fp",
        boundary_epoch=1000,
        cycle_id=1,
    )
    assert (
        resolve_cached_prediction(
            orch,
            "R_10",
            at_boundary=False,
            boundary_epoch=1000,
            cycle_id=2,
        )
        is None
    )
    assert (
        resolve_cached_prediction(
            orch,
            "R_10",
            at_boundary=True,
            tensor_fingerprint=b"fp",
            boundary_epoch=1000,
            cycle_id=1,
        )
        is entry
    )
    orch._dl_prediction_cache["R_10"]["cycle_id"] = object()
    assert (
        resolve_cached_prediction(
            orch,
            "R_10",
            at_boundary=False,
            boundary_epoch=1000,
            cycle_id=1,
        )
        is None
    )


def test_resolve_cached_prediction_requires_fingerprint_on_boundary():
    orch = SimpleNamespace(_dl_prediction_cache={})
    store_prediction_cache(orch, "R_10", {"direction": "CALL"}, tensor_fingerprint=b"x", boundary_epoch=1)
    assert resolve_cached_prediction(orch, "R_10", at_boundary=True, tensor_fingerprint=None) is None


def test_resolve_cached_prediction_ignores_invalid_entry():
    orch = SimpleNamespace(_dl_prediction_cache={"R_10": {"entry": "invalid"}})
    assert resolve_cached_prediction(orch, "R_10", at_boundary=False) is None


def test_prediction_cache_initializes_on_orchestrator():
    orch = SimpleNamespace()
    cache = prediction_cache(orch)
    assert cache is orch._dl_prediction_cache
    assert cache == prediction_cache(orch)
