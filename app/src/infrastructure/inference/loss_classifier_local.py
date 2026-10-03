"""Executor in-process local para o Loss-Classifier sem salto de rede HTTP."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from aether_paths import repo_path


logger = logging.getLogger("AETH")

_CANDIDATE_PATHS = (
    repo_path("infra", "docker", "loss-models", "loss_bootstrap_live64.pkl"),
    repo_path("data", "dl", "loss_classifier.pkl"),
    repo_path("infra", "docker", "loss-models", "loss_classifier.joblib"),
)


class LocalLossClassifier:
    """Executa a inferencia do classificador de perdas diretamente no processo do host."""

    def __init__(self, model_path: Path | str | None = None) -> None:
        """Inicializa classificador procurando artefatos serializados."""
        self._model_path = Path(model_path) if model_path is not None else None
        self._model: Any | None = None
        self._is_ready: bool = False
        self._n_train: int = 0
        self._load_model()

    def _load_model(self) -> None:
        """Carrega booster em memoria RAM persistente."""
        paths = [self._model_path] if self._model_path else list(_CANDIDATE_PATHS)
        for p in paths:
            if p is not None and p.is_file():
                try:
                    loaded = joblib.load(p)
                    if isinstance(loaded, dict):
                        self._model = loaded.get("model") or loaded.get("clf")
                        self._n_train = int(loaded.get("n_train", 64))
                    else:
                        self._model = loaded
                        self._n_train = 64
                    self._is_ready = self._model is not None
                    if self._is_ready:
                        logger.info("LOSS_LOCAL: Carregado modelo local de %s", p)
                        break
                except Exception as exc:
                    logger.debug("LOSS_LOCAL: Falha ao carregar %s: %s", p, exc)

    @property
    def is_ready(self) -> bool:
        """Indica se o modelo esta pronto para predicao."""
        return self._is_ready

    def predict(self, feature_vector: list[float] | np.ndarray) -> tuple[float, float, int]:
        """Calcula P(loss), P(eff) e n_train em menos de 0.1 ms."""
        if not self._is_ready or self._model is None:
            return 0.50, 0.50, self._n_train

        arr = np.asarray(feature_vector, dtype=np.float32).reshape(1, -1)
        try:
            if hasattr(self._model, "predict_proba"):
                probs = self._model.predict_proba(arr)
                p_loss = float(probs[0][1]) if probs.shape[1] > 1 else float(probs[0][0])
            elif hasattr(self._model, "predict"):
                pred = float(self._model.predict(arr)[0])
                p_loss = 1.0 / (1.0 + np.exp(-pred))
            else:
                p_loss = 0.50
        except Exception:
            p_loss = 0.50

        p_loss = max(0.0, min(1.0, p_loss))
        p_eff = p_loss
        return p_loss, p_eff, self._n_train
