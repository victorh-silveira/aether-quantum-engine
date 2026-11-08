"""Testes unitarios para o motor de inferencia ONNX do TCN."""

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np

import src.infrastructure.inference.tcn_onnx_infer as infer_mod
from src.infrastructure.inference.tcn_onnx_infer import TcnOnnxInferenceEngine


def test_tcn_onnx_infer_non_existent_path():
    engine = TcnOnnxInferenceEngine(Path("non/existent/model.onnx"))
    assert not engine.is_ready
    assert engine.predict(np.zeros((1, 32, 10), dtype=np.float32)) is None


def test_tcn_onnx_infer_none_path():
    engine = TcnOnnxInferenceEngine(None)
    assert not engine.is_ready
    assert engine.predict(np.zeros((32, 10), dtype=np.float32)) is None


def test_tcn_onnx_infer_is_available_returns_bool():
    available = TcnOnnxInferenceEngine.is_onnxruntime_available()
    assert isinstance(available, bool)


def test_tcn_onnx_infer_success_2d_and_3d(tmp_path, monkeypatch):
    model_file = tmp_path / "model.onnx"
    model_file.write_bytes(b"dummy")

    class MockInferenceSession:
        def __init__(self, path: Any, sess_options: Any = None, providers: Any = None):
            _ = (path, sess_options, providers)

        def get_inputs(self):
            return [SimpleNamespace(name="input_features")]

        def get_outputs(self):
            return [SimpleNamespace(name="logits")]

        def run(self, output_names: Any, feed_dict: Any) -> list[np.ndarray]:
            _ = (output_names, feed_dict)
            return [np.array([0.5], dtype=np.float32)]

    class MockOrt:
        class SessionOptions:
            def __init__(self):
                self.intra_op_num_threads = 0
                self.inter_op_num_threads = 0
                self.graph_optimization_level = None

        class GraphOptimizationLevel:
            ORT_ENABLE_ALL = 1

        InferenceSession = MockInferenceSession

        @staticmethod
        def get_available_providers():
            return ["CPUExecutionProvider"]

    monkeypatch.setattr(infer_mod, "_ORT_AVAILABLE", True)
    monkeypatch.setattr(infer_mod, "ort", MockOrt)

    engine = TcnOnnxInferenceEngine(model_file)
    assert engine.is_ready is True

    prob_2d = engine.predict(np.zeros((32, 14), dtype=np.float32))
    assert prob_2d is not None
    assert 0.0 <= prob_2d <= 1.0

    prob_3d = engine.predict(np.zeros((1, 32, 14), dtype=np.float32))
    assert prob_3d is not None


def test_tcn_onnx_infer_session_exception(tmp_path, monkeypatch):
    model_file = tmp_path / "error_model.onnx"
    model_file.write_bytes(b"dummy")

    class BrokenOrt:
        @staticmethod
        def get_available_providers():
            raise RuntimeError("Falha no provedor")

    monkeypatch.setattr(infer_mod, "_ORT_AVAILABLE", True)
    monkeypatch.setattr(infer_mod, "ort", BrokenOrt)

    engine = TcnOnnxInferenceEngine(model_file)
    assert engine.is_ready is False


def test_tcn_onnx_infer_predict_exception(tmp_path, monkeypatch):
    model_file = tmp_path / "model.onnx"
    model_file.write_bytes(b"dummy")

    class FailingSession:
        def __init__(self, path: Any, sess_options: Any = None, providers: Any = None):
            _ = (path, sess_options, providers)

        def get_inputs(self):
            return [SimpleNamespace(name="input_features")]

        def get_outputs(self):
            return [SimpleNamespace(name="logits")]

        def run(self, output_names: Any, feed_dict: Any) -> list[np.ndarray]:
            _ = (output_names, feed_dict)
            raise RuntimeError("Erro interno no grafo C++")

    class MockOrt:
        class SessionOptions:
            pass

        class GraphOptimizationLevel:
            ORT_ENABLE_ALL = 1

        InferenceSession = FailingSession

        @staticmethod
        def get_available_providers():
            return ["CPUExecutionProvider"]

    monkeypatch.setattr(infer_mod, "_ORT_AVAILABLE", True)
    monkeypatch.setattr(infer_mod, "ort", MockOrt)

    engine = TcnOnnxInferenceEngine(model_file)
    assert engine.is_ready is True
    assert engine.predict(np.zeros((32, 14), dtype=np.float32)) is None


def test_tcn_onnx_infer_reload_with_module(monkeypatch):
    mock_ort = ModuleType("onnxruntime")
    monkeypatch.setitem(sys.modules, "onnxruntime", mock_ort)
    importlib.reload(infer_mod)
    assert infer_mod.TcnOnnxInferenceEngine.is_onnxruntime_available() is True

    monkeypatch.delitem(sys.modules, "onnxruntime", raising=False)
    importlib.reload(infer_mod)
