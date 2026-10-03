"""Motor de inferencia de ultra-baixa latencia para TCN via ONNX Runtime."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

import numpy as np


logger = logging.getLogger("AETH")

try:
    ort = importlib.import_module("onnxruntime")
    _ORT_AVAILABLE = True
except ImportError:
    ort = None
    _ORT_AVAILABLE = False


class TcnOnnxInferenceEngine:
    """Sessao de inferencia C++ ONNX com quantizacao e provedores acelerados."""

    def __init__(self, onnx_model_path: Path | str | None = None) -> None:
        """Inicializa sessao ONNX identificando provedores de execucao compativeis."""
        self._model_path = Path(onnx_model_path) if onnx_model_path is not None else None
        self._session: Any | None = None
        self._input_name: str | None = None
        self._output_name: str | None = None
        self._is_ready: bool = False

        if _ORT_AVAILABLE and ort is not None and self._model_path is not None and self._model_path.exists():
            try:
                available_providers = ort.get_available_providers()
                preferred = [
                    p
                    for p in ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
                    if p in available_providers
                ]
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 1
                opts.inter_op_num_threads = 1
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self._session = ort.InferenceSession(str(self._model_path), sess_options=opts, providers=preferred)
                self._input_name = self._session.get_inputs()[0].name
                self._output_name = self._session.get_outputs()[0].name
                self._is_ready = True
            except Exception as exc:
                logger.debug("ONNX: Erro ao instanciar InferenceSession: %s", exc)
                self._is_ready = False
                self._session = None

    @staticmethod
    def is_onnxruntime_available() -> bool:
        """Informa se a biblioteca onnxruntime esta instalada no ambiente."""
        return _ORT_AVAILABLE

    @property
    def is_ready(self) -> bool:
        """Indica se a sessao de inferencia compilada esta ativa."""
        return self._is_ready

    def predict(self, feature_tensor: np.ndarray) -> float | None:
        """Executa a inferencia batch=1 com zero-overhead do interpretador Python."""
        if not self._is_ready or self._session is None or self._input_name is None:
            return None

        arr = np.asarray(feature_tensor, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=0)

        try:
            outputs = self._session.run([self._output_name], {self._input_name: arr})
            logits = outputs[0]
            val = float(logits.squeeze())
            prob = 1.0 / (1.0 + np.exp(-val)) if np.isfinite(val) else 0.50
            return float(np.clip(prob, 0.0, 1.0))
        except Exception as exc:
            logger.debug("ONNX: Erro na inferencia: %s", exc)
            return None
