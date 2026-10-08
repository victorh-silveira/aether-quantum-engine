"""Contrato schema/hash dos sidecars meta/loss e fallback sem edge 0.0 falso."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

from src.application.services.loss_classifier_features import LOSS_FEATURE_NAMES, loss_feature_schema_hash
from src.application.services.meta_classifier_features import meta_classifier_column_names, meta_feature_schema_hash
from src.application.services.ml_schema_hash import canonical_schema_hash
from src.infrastructure.inference.meta_classifier_types import parse_meta_predict_response


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "infra" / "docker").is_dir():
            return parent
    return Path.cwd()


def _ensure_ml_common() -> None:
    root = str(_repo_root() / "infra" / "docker")
    if root not in sys.path:
        sys.path.insert(0, root)


def test_motor_and_sidecar_schema_hashes_match():
    _ensure_ml_common()
    from ml_common import names as sidecar_names
    from ml_common.schema import canonical_schema_hash as sidecar_hash

    assert tuple(LOSS_FEATURE_NAMES) == tuple(sidecar_names.LOSS_FEATURE_NAMES)
    assert tuple(meta_classifier_column_names()) == tuple(sidecar_names.META_FEATURE_NAMES)
    assert loss_feature_schema_hash() == sidecar_hash(sidecar_names.LOSS_FEATURE_NAMES)
    assert meta_feature_schema_hash() == sidecar_hash(sidecar_names.META_FEATURE_NAMES)
    assert canonical_schema_hash(LOSS_FEATURE_NAMES) == loss_feature_schema_hash()


def test_schema_hash_payload_ok_and_mismatch():
    _ensure_ml_common()
    from ml_common.schema import bundle_schema_hash_ok, request_schema_hash_ok

    assert bundle_schema_hash_ok({}, "abc") is True
    assert bundle_schema_hash_ok({"schema_hash": "abc"}, "abc") is True
    assert bundle_schema_hash_ok({"schema_hash": "zzz"}, "abc") is False
    assert request_schema_hash_ok(None, "abc") is True
    assert request_schema_hash_ok("abc", "abc") is True
    assert request_schema_hash_ok("zzz", "abc") is False


def test_validate_feature_vector_rejects_nan_and_wrong_dim():
    _ensure_ml_common()
    from ml_common.schema import validate_feature_vector

    assert validate_feature_vector([1.0, 2.0], 2) == [1.0, 2.0]
    with pytest.raises(ValueError, match="elementos"):
        validate_feature_vector([1.0], 2)
    with pytest.raises(ValueError, match="nao finito"):
        validate_feature_vector([1.0, math.nan], 2)
    with pytest.raises(ValueError, match="lista"):
        validate_feature_vector("x", 1)


def test_contract_id_dedupe_caps_and_ignores_blank():
    _ensure_ml_common()
    from ml_common.learn_ids import ContractIdDedupe

    bag = ContractIdDedupe(max_ids=2)
    assert bag.seen_or_add("") is False
    assert bag.seen_or_add("a") is False
    assert bag.seen_or_add("a") is True
    assert bag.seen_or_add("b") is False
    assert bag.seen_or_add("c") is False
    assert bag.seen_or_add("a") is False


def test_loss_recency_and_degenerate_quality():
    sidecar = str(_repo_root() / "infra" / "docker" / "loss-classifier")
    if sidecar not in sys.path:
        sys.path.insert(0, sidecar)
    _ensure_ml_common()
    import runtime as loss_runtime

    young = loss_runtime.recency_sample_weights(8, half_life=32, mature=False)
    assert young is None
    weights = loss_runtime.recency_sample_weights(4, half_life=2, mature=True)
    assert weights is not None
    assert weights[-1] == pytest.approx(1.0)
    assert weights[0] < weights[-1]
    assert loss_runtime.is_degenerate_quality(collapsed=True, cal_ece=0.01, n_train=8) is True
    assert loss_runtime.is_degenerate_quality(collapsed=False, cal_ece=0.40, n_train=32, bootstrap=True) is False
    assert loss_runtime.is_degenerate_quality(collapsed=False, cal_ece=0.40, n_train=32) is True
    assert loss_runtime.is_degenerate_quality(collapsed=False, cal_ece=0.40, n_train=8) is False
    assert loss_runtime.is_degenerate_quality(collapsed=False, cal_ece=0.10, n_train=64) is False


def test_parse_meta_predict_omits_edge_when_not_applied():
    parsed = parse_meta_predict_response({"meta_applied": False})
    assert parsed["meta_applied"] is False
    assert parsed["predicted_payoff_edge"] is None
    parsed_null = parse_meta_predict_response({"meta_applied": False, "predicted_payoff_edge": None})
    assert parsed_null["predicted_payoff_edge"] is None
    with pytest.raises(KeyError):
        parse_meta_predict_response({"meta_applied": True})


def test_atomic_persist_roundtrip(tmp_path: Path):
    _ensure_ml_common()
    from ml_common.persist import atomic_joblib_dump, atomic_pickle_dump

    joblib_path = tmp_path / "a.pkl"
    pickle_path = tmp_path / "b.pkl"
    atomic_joblib_dump({"n": 1}, joblib_path)
    atomic_pickle_dump({"n": 2}, pickle_path)
    assert joblib_path.is_file()
    assert pickle_path.is_file()
