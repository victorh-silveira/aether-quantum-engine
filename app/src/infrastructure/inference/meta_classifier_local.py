"""Executor in-process local para o meta-classificador LightGBM sem overhead HTTP."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from aether_paths import repo_path
from src.application.services.meta_classifier_cross_symbol import META_FEATURE_DIM
from src.infrastructure.inference.meta_classifier_types import (
    MetaPredictRequest,
    MetaPredictResponse,
    classify_edge_expectancy_from_payoff,
)


_CANDIDATE_PATHS = (
    repo_path("infra", "docker", "meta-models", "meta_lgbm.pkl"),
    repo_path("data", "dl", "meta_candidate.joblib"),
    repo_path("data", "meta", "meta_lgbm.pkl"),
)

PAYOFF_CLAMP_MIN = -1.0
PAYOFF_CLAMP_MAX = 0.85


class LocalMetaClassifier:
    """Carrega e executa o modelo LightGBM localmente no processo do host."""

    def __init__(self, model_path: Path | str | None = None):
        self._model_path = Path(model_path) if model_path is not None else None
        self._model: Any | None = None
        self._label_scale: float = 1.0
        self._is_ready: bool = False
        self._load_lock = asyncio.Lock()

    def is_ready(self) -> bool:
        """Indica se o modelo local esta carregado e pronto para predicoes."""
        return self._is_ready and self._model is not None

    def _resolve_existing_model_path(self) -> Path | None:
        """Localiza o primeiro artefato de modelo existente nos caminhos candidatos."""
        if self._model_path is not None:
            return self._model_path if self._model_path.is_file() else None
        for candidate in _CANDIDATE_PATHS:
            if candidate.is_file():
                return candidate
        return None

    def load_model_sync(self) -> bool:
        """Carrega sincronamente o booster LightGBM em memoria."""
        path = self._resolve_existing_model_path()
        if path is None:
            self._is_ready = False
            return False
        try:
            artifact = joblib.load(path)
            if isinstance(artifact, dict) and "model" in artifact:
                self._model = artifact["model"]
                self._label_scale = float(artifact.get("label_scale", 1.0) or 1.0)
            else:
                self._model = artifact
                self._label_scale = 1.0
            self._is_ready = self._model is not None
            return self._is_ready
        except Exception:
            self._model = None
            self._is_ready = False
            return False

    async def ensure_ready(self) -> bool:
        """Garante que o modelo esteja carregado antes da inferencia."""
        if self.is_ready():
            return True
        async with self._load_lock:
            if self.is_ready():
                return True
            return await asyncio.to_thread(self.load_model_sync)

    def _predict_raw(self, feature_vector: list[float]) -> float:
        """Executa predicao do modelo em vetor 23D com scale e clamp de payoff."""
        if not self._model or len(feature_vector) != META_FEATURE_DIM:
            raise ValueError("Modelo nao carregado ou dimensao de feature incorreta")
        inp = np.asarray([feature_vector], dtype=np.float64)
        pred = float(self._model.predict(inp)[0])
        scaled = pred * self._label_scale
        return max(PAYOFF_CLAMP_MIN, min(PAYOFF_CLAMP_MAX, scaled))

    async def predict_meta_local(
        self,
        request: MetaPredictRequest,
        *,
        fallback_score: float = 0.5,
    ) -> MetaPredictResponse:
        """Executa inferencia in-process do LightGBM sem salto de rede HTTP."""
        _ = fallback_score
        ready = await self.ensure_ready()
        if not ready or self._model is None:
            return {
                "predicted_payoff_edge": None,
                "meta_applied": False,
                "edge_expectancy": "LOSS_EXPECTED",
            }
        try:
            vec = [float(v) for v in request["feature_vector"]]
            edge = await asyncio.to_thread(self._predict_raw, vec)
            expectancy = classify_edge_expectancy_from_payoff(edge)
            return {
                "predicted_payoff_edge": edge,
                "meta_applied": True,
                "edge_expectancy": expectancy,
            }
        except Exception:
            return {
                "predicted_payoff_edge": None,
                "meta_applied": False,
                "edge_expectancy": "LOSS_EXPECTED",
            }


_GLOBAL_LOCAL_META = LocalMetaClassifier()


def get_global_local_meta_classifier() -> LocalMetaClassifier:
    """Retorna singleton do meta-classificador local in-process."""
    return _GLOBAL_LOCAL_META
