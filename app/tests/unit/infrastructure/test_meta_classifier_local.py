"""Testes unitarios para o executor in-process local meta_classifier_local."""

from unittest.mock import patch

import numpy as np
import pytest

import src.infrastructure.inference.meta_classifier_local as meta_local_mod
from src.infrastructure.inference.meta_classifier_local import (
    PAYOFF_CLAMP_MAX,
    PAYOFF_CLAMP_MIN,
    LocalMetaClassifier,
    get_global_local_meta_classifier,
)


class MockLgbmModel:
    """Mock serializavel de booster LightGBM para testes unitarios."""

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Retorna previsao fixa compativel com o formato numpy."""
        return np.array([0.15] * len(x))


@pytest.fixture
def mock_lgbm_model():
    """Fornece instancia serializavel do modelo mock."""
    return MockLgbmModel()


def test_local_meta_init_state():
    classifier = LocalMetaClassifier(model_path="/non/existent/path.pkl")
    assert classifier.is_ready() is False


def test_local_meta_load_missing_file():
    classifier = LocalMetaClassifier(model_path="/non/existent/path.pkl")
    assert classifier.load_model_sync() is False
    assert classifier.is_ready() is False


def test_local_meta_resolve_candidate_paths(tmp_path, mock_lgbm_model, monkeypatch):
    import joblib

    p = tmp_path / "candidate_model.pkl"
    joblib.dump({"model": mock_lgbm_model}, p)

    monkeypatch.setattr(meta_local_mod, "_CANDIDATE_PATHS", (tmp_path / "missing.pkl", p))
    classifier = LocalMetaClassifier(model_path=None)
    resolved = classifier._resolve_existing_model_path()
    assert resolved == p

    monkeypatch.setattr(meta_local_mod, "_CANDIDATE_PATHS", (tmp_path / "missing1.pkl", tmp_path / "missing2.pkl"))
    classifier_none = LocalMetaClassifier(model_path=None)
    assert classifier_none._resolve_existing_model_path() is None


def test_local_meta_load_dict_artifact(tmp_path, mock_lgbm_model):
    import joblib

    p = tmp_path / "test_meta.pkl"
    joblib.dump({"model": mock_lgbm_model, "label_scale": 1.2}, p)

    classifier = LocalMetaClassifier(model_path=p)
    assert classifier.load_model_sync() is True
    assert classifier.is_ready() is True


def test_local_meta_load_raw_artifact_and_load_exception(tmp_path, mock_lgbm_model):
    import joblib

    p_raw = tmp_path / "raw_meta.pkl"
    joblib.dump(mock_lgbm_model, p_raw)

    classifier_raw = LocalMetaClassifier(model_path=p_raw)
    assert classifier_raw.load_model_sync() is True
    assert classifier_raw.is_ready() is True

    p_broken = tmp_path / "broken.pkl"
    p_broken.write_bytes(b"corrupt")

    with patch("joblib.load", side_effect=RuntimeError("Corrompido")):
        classifier_broken = LocalMetaClassifier(model_path=p_broken)
        assert classifier_broken.load_model_sync() is False
        assert classifier_broken.is_ready() is False


def test_local_meta_predict_raw_dimension_check(tmp_path, mock_lgbm_model):
    import joblib

    p = tmp_path / "test_meta.pkl"
    joblib.dump({"model": mock_lgbm_model}, p)

    classifier = LocalMetaClassifier(model_path=p)
    classifier.load_model_sync()

    with pytest.raises(ValueError, match="dimensao de feature"):
        classifier._predict_raw([1.0, 2.0])

    vector_23d = [0.1] * 23
    val = classifier._predict_raw(vector_23d)
    assert isinstance(val, float)
    assert PAYOFF_CLAMP_MIN <= val <= PAYOFF_CLAMP_MAX


@pytest.mark.asyncio
async def test_local_meta_ensure_ready_branches(tmp_path, mock_lgbm_model):
    import joblib

    p = tmp_path / "test_meta.pkl"
    joblib.dump({"model": mock_lgbm_model}, p)

    classifier = LocalMetaClassifier(model_path=p)
    classifier.load_model_sync()
    assert await classifier.ensure_ready() is True

    classifier2 = LocalMetaClassifier(model_path=p)
    assert await classifier2.ensure_ready() is True


@pytest.mark.asyncio
async def test_local_meta_ensure_ready_inside_lock():
    classifier = LocalMetaClassifier(model_path=None)
    with patch.object(classifier, "is_ready", side_effect=[False, True]):
        assert await classifier.ensure_ready() is True


@pytest.mark.asyncio
async def test_local_meta_predict_meta_local_success(tmp_path, mock_lgbm_model):
    import joblib

    p = tmp_path / "test_meta.pkl"
    joblib.dump({"model": mock_lgbm_model, "label_scale": 1.0}, p)

    classifier = LocalMetaClassifier(model_path=p)
    req = {
        "symbol": "1HZ75V",
        "tcn_probability": 0.60,
        "direction": "CALL",
        "feature_vector": [0.0] * 23,
    }
    resp = await classifier.predict_meta_local(req)
    assert resp["meta_applied"] is True
    assert resp["predicted_payoff_edge"] == pytest.approx(0.15)
    assert resp["edge_expectancy"] == "WIN_EXPECTED"


@pytest.mark.asyncio
async def test_local_meta_predict_meta_local_when_not_ready():
    classifier = LocalMetaClassifier(model_path="/invalid/path.pkl")
    req = {
        "symbol": "1HZ75V",
        "tcn_probability": 0.60,
        "direction": "CALL",
        "feature_vector": [0.0] * 23,
    }
    resp = await classifier.predict_meta_local(req)
    assert resp["meta_applied"] is False
    assert resp["predicted_payoff_edge"] is None
    assert resp["edge_expectancy"] == "LOSS_EXPECTED"


@pytest.mark.asyncio
async def test_local_meta_predict_meta_local_exception(tmp_path, mock_lgbm_model):
    import joblib

    p = tmp_path / "test_meta.pkl"
    joblib.dump({"model": mock_lgbm_model}, p)

    classifier = LocalMetaClassifier(model_path=p)
    req = {
        "symbol": "1HZ75V",
        "tcn_probability": 0.60,
        "direction": "CALL",
        "feature_vector": ["invalid_feature_entry"],
    }
    resp = await classifier.predict_meta_local(req)
    assert resp["meta_applied"] is False
    assert resp["predicted_payoff_edge"] is None


def test_get_global_local_meta_classifier():
    instance = get_global_local_meta_classifier()
    assert isinstance(instance, LocalMetaClassifier)
