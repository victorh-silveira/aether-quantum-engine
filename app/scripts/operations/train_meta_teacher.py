"""Inferencia do teacher TCN a partir de checkpoints DL para o treino meta."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.operations.train_meta_data import OhlcBundle
from src.application.services.deep_learning.dl_calibration import CalibratorState, apply_calibrator
from src.application.services.deep_learning.dl_device import place_model, resolve_torch_device
from src.application.services.deep_learning.dl_feature_build import precompute_price_series
from src.application.services.deep_learning.dl_feature_matrix import build_feature_matrix
from src.application.services.deep_learning.dl_model_checkpoint import load_model_checkpoint
from src.application.services.deep_learning.model import _model_raw_prob, normalize_sequences


logger = logging.getLogger("META_TRAIN")

TEACHER_BATCH_SIZE = 256
TEACHER_PROB_FLOOR = 0.05
TEACHER_PROB_CEIL = 0.95


def _clip_teacher_prob(raw: float, calibrator: CalibratorState | None) -> float:
    prob = float(apply_calibrator(raw, calibrator)) if calibrator is not None else float(raw)
    return float(np.clip(prob, TEACHER_PROB_FLOOR, TEACHER_PROB_CEIL))


def infer_teacher_probs_for_bundle(
    bundle: OhlcBundle,
    *,
    model: Any,
    norm_stats: Any,
    lookback: int,
    calibrator: CalibratorState | None = None,
    batch_size: int = TEACHER_BATCH_SIZE,
) -> np.ndarray:
    """Gera serie de P(CALL) alinhada ao OHLC do bundle via rolling window TCN."""
    closes = np.asarray(bundle.closes, dtype=np.float64)
    n = int(len(closes))
    probs = np.full(n, 0.5, dtype=np.float32)
    lb = max(1, int(lookback))
    if n < lb:
        return probs
    series = precompute_price_series(
        closes,
        granularity=int(bundle.granularity),
        symbol=str(bundle.symbol),
        open_=bundle.open_,
        high=bundle.high,
        low=bundle.low,
    )
    feature_matrix = build_feature_matrix(series)
    ends = list(range(lb - 1, n))
    raw_chunks: list[np.ndarray] = []
    step = max(1, int(batch_size))
    for start in range(0, len(ends), step):
        chunk_ends = ends[start : start + step]
        batch = np.stack(
            [feature_matrix[end - lb + 1 : end + 1] for end in chunk_ends],
            axis=0,
        ).astype(np.float32)
        feat = normalize_sequences(batch, norm_stats)
        raw_chunks.append(_model_raw_prob(model, feat))
    raw = np.concatenate(raw_chunks, axis=0) if raw_chunks else np.empty((0,), dtype=np.float32)
    for idx, end in enumerate(ends):
        probs[end] = _clip_teacher_prob(float(raw[idx]), calibrator)
    return probs


def infer_teacher_probs_from_checkpoints(
    bundles: list[OhlcBundle],
    *,
    model_path_template: str,
    dl_params: dict[str, Any] | None = None,
    batch_size: int = TEACHER_BATCH_SIZE,
) -> dict[str, np.ndarray]:
    """Carrega checkpoints por simbolo e infere probs teacher alinhadas aos bundles."""
    loaded: dict[str, np.ndarray] = {}
    params = dl_params if isinstance(dl_params, dict) else {}
    device = resolve_torch_device(params, kind="inference")
    for bundle in bundles:
        path = Path(str(model_path_template).format(symbol=bundle.symbol))
        if not path.is_file():
            logger.debug("teacher checkpoint ausente: %s", path)
            continue
        try:
            ckpt = load_model_checkpoint(path, params=params)
        except Exception as exc:
            logger.warning("teacher checkpoint invalido %s: %s", path, exc)
            continue
        if ckpt is None:
            logger.warning("teacher checkpoint incompativel: %s", path)
            continue
        model, norm_stats, _epoch, calibrator, lookback, *_rest = ckpt
        place_model(model, device)
        model.eval()
        try:
            probs = infer_teacher_probs_for_bundle(
                bundle,
                model=model,
                norm_stats=norm_stats,
                lookback=int(lookback),
                calibrator=calibrator,
                batch_size=batch_size,
            )
        except Exception as exc:
            logger.warning("teacher inferencia falhou %s: %s", bundle.symbol, exc)
            continue
        finally:
            place_model(model, torch.device("cpu"))
        if int(probs.size) != int(len(bundle.closes)):
            continue
        loaded[str(bundle.symbol)] = probs.astype(np.float32)
        logger.info(
            "META_TRAIN: teacher TCN inferido | %s | n=%d | lookback=%d | path=%s",
            bundle.symbol,
            int(probs.size),
            int(lookback),
            path,
        )
    return loaded


def load_teacher_probs_from_checkpoints(
    symbols: list[str],
    *,
    model_path_template: str,
    lookback: int,
    repo_root: Path,
    bundles: list[OhlcBundle] | None = None,
    dl_params: dict[str, Any] | None = None,
    batch_size: int = TEACHER_BATCH_SIZE,
) -> dict[str, np.ndarray]:
    """API de compatibilidade: com bundles, infere; sem bundles, tenta arrays embutidos."""
    _ = lookback
    if bundles:
        selected = [bundle for bundle in bundles if str(bundle.symbol) in set(symbols)]
        return infer_teacher_probs_from_checkpoints(
            selected,
            model_path_template=model_path_template,
            dl_params=dl_params,
            batch_size=batch_size,
        )
    loaded: dict[str, np.ndarray] = {}
    for symbol in symbols:
        path = Path(str(model_path_template).format(symbol=symbol))
        if not path.is_file():
            continue
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            logger.debug("teacher checkpoint skip %s: %s", path, exc)
            continue
        probs = payload.get("teacher_probs") if isinstance(payload, dict) else None
        if probs is None and isinstance(payload, dict):
            probs = payload.get("val_probs")
        if probs is None:
            continue
        arr = np.asarray(probs, dtype=np.float32).reshape(-1)
        if arr.size > 0:
            loaded[str(symbol)] = arr
    _ = repo_root
    return loaded
