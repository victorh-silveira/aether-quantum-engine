"""Testes unitarios para o Loss-Classifier local in-process."""

from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.infrastructure.inference.loss_classifier_local import LocalLossClassifier


def test_loss_classifier_local_missing_path():
    """Valida fallback seguro quando o arquivo do modelo nao existe."""
    clf = LocalLossClassifier(model_path=Path("non/existent/model.pkl"))
    assert not clf.is_ready

    p_loss, p_eff, n_train = clf.predict([0.1] * 24)
    assert p_loss == 0.50
    assert p_eff == 0.50
    assert n_train == 0


def test_loss_classifier_local_dummy_booster():
    """Valida inferencia com predict_proba multiclasse/binario padrao."""

    class DummyModel:
        def predict_proba(self, _x):
            return np.array([[0.3, 0.7]])

    clf = LocalLossClassifier()
    clf._model = DummyModel()
    clf._is_ready = True
    clf._n_train = 64

    p_loss, p_eff, n_train = clf.predict([0.0] * 24)
    assert p_loss == 0.70
    assert p_eff == 0.70
    assert n_train == 64


def test_loss_classifier_local_load_dict_with_model_and_ntrain(tmp_path):
    """Valida carregamento de dicionario contendo chave model e n_train."""

    class DummyModel:
        def predict_proba(self, _x):
            return np.array([[0.2, 0.8]])

    model_file = tmp_path / "model.joblib"
    model_file.touch()

    with patch("joblib.load", return_value={"model": DummyModel(), "n_train": 128}):
        clf = LocalLossClassifier(model_path=model_file)
        assert clf.is_ready
        p_loss, p_eff, n_train = clf.predict([0.0] * 24)
        assert p_loss == 0.80
        assert n_train == 128


def test_loss_classifier_local_load_dict_with_clf(tmp_path):
    """Valida carregamento de dicionario contendo chave clf sem n_train."""

    class DummyModel:
        def predict_proba(self, _x):
            return np.array([[0.6]])

    model_file = tmp_path / "model_clf.joblib"
    model_file.touch()

    with patch("joblib.load", return_value={"clf": DummyModel()}):
        clf = LocalLossClassifier(model_path=model_file)
        assert clf.is_ready
        p_loss, p_eff, n_train = clf.predict([0.0] * 24)
        assert p_loss == 0.60
        assert n_train == 64


def test_loss_classifier_local_load_raw_model_instance(tmp_path):
    """Valida carregamento direto de instancia de modelo sem wrapper dict."""

    class RawModel:
        def predict(self, _x):
            return np.array([0.0])

    model_file = tmp_path / "raw_model.pkl"
    model_file.touch()

    with patch("joblib.load", return_value=RawModel()):
        clf = LocalLossClassifier(model_path=model_file)
        assert clf.is_ready
        p_loss, p_eff, n_train = clf.predict([0.0] * 24)
        assert p_loss == 0.50
        assert n_train == 64


def test_loss_classifier_local_load_corrupted_file(tmp_path):
    """Valida resiliencia quando joblib.load lanca excecao."""
    corrupted_file = tmp_path / "corrupted.pkl"
    corrupted_file.touch()

    with patch("joblib.load", side_effect=RuntimeError("Corrompido")):
        clf = LocalLossClassifier(model_path=corrupted_file)
        assert not clf.is_ready


def test_loss_classifier_local_predict_no_methods():
    """Valida modelo sem predict_proba nem predict retornando 0.50."""

    class EmptyModel:
        pass

    clf = LocalLossClassifier()
    clf._model = EmptyModel()
    clf._is_ready = True

    p_loss, p_eff, _ = clf.predict([0.0] * 24)
    assert p_loss == 0.50
    assert p_eff == 0.50


def test_loss_classifier_local_predict_exception():
    """Valida tratamento quando metodo de predicao lanca excecao."""

    class FailingModel:
        def predict_proba(self, _x):
            raise ValueError("Erro de computacao")

    clf = LocalLossClassifier()
    clf._model = FailingModel()
    clf._is_ready = True

    p_loss, p_eff, _ = clf.predict([0.0] * 24)
    assert p_loss == 0.50
    assert p_eff == 0.50
