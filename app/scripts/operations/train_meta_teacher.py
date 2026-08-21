"""Inferencia do teacher TCN a partir de checkpoints DL para o treino meta."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from aether_paths import REPO_ROOT
from scripts.operations.train_meta_data import OhlcBundle
from src.application.services.deep_learning.dl_calibration import CalibratorState, apply_calibrator
from src.application.services.deep_learning.dl_device import place_model, resolve_torch_device
from src.application.services.deep_learning.dl_feature_build import precompute_price_series
from src.application.services.deep_learning.dl_feature_matrix import build_feature_matrix
from src.application.services.deep_learning.dl_model_checkpoint import load_model_checkpoint
from src.application.services.deep_learning.model import _model_raw_prob, normalize_sequences


logger = logging.getLogger("AETH.meta")

TEACHER_BATCH_SIZE = 256
TEACHER_PROB_FLOOR = 0.05
TEACHER_PROB_CEIL = 0.95
TEACHER_COLLAPSE_STD = 1e-3
TEACHER_MIN_STD_RATIO = 0.5
TEACHER_EXPAND_STD_TRIGGER = 0.02
TEACHER_EXPAND_SCALE = 0.35


def expand_teacher_conviction(
    probs: np.ndarray,
    *,
    scale: float = TEACHER_EXPAND_SCALE,
) -> np.ndarray:
    x = np.asarray(probs, dtype=np.float64)
    if x.size == 0:
        return x.astype(np.float32)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) + 1e-9
    z = (x - med) / (1.4826 * mad)
    out = 0.5 + float(scale) * np.tanh(z)
    return np.clip(out, TEACHER_PROB_FLOOR, TEACHER_PROB_CEIL).astype(np.float32)


def _calibrate_teacher_array(
    raw: np.ndarray,
    calibrator: CalibratorState | None,
) -> tuple[np.ndarray, str]:
    raw_arr = np.asarray(raw, dtype=np.float64).reshape(-1)
    if raw_arr.size == 0:
        return raw_arr.astype(np.float32), "empty"
    if calibrator is None:
        chosen = np.clip(raw_arr, TEACHER_PROB_FLOOR, TEACHER_PROB_CEIL)
        source = "raw"
    else:
        calibrated = np.asarray(
            [float(apply_calibrator(float(v), calibrator)) for v in raw_arr],
            dtype=np.float64,
        )
        raw_std = float(np.std(raw_arr))
        cal_std = float(np.std(calibrated))
        if cal_std < TEACHER_COLLAPSE_STD or cal_std < TEACHER_MIN_STD_RATIO * max(raw_std, 1e-12):
            logger.warning(
                "META_TRAIN: calibrador teacher colapsou variancia (raw_std=%.6f cal_std=%.6f); usando probs raw.",
                raw_std,
                cal_std,
            )
            chosen = np.clip(raw_arr, TEACHER_PROB_FLOOR, TEACHER_PROB_CEIL)
            source = "raw_collapsed_calibrator"
        else:
            chosen = np.clip(calibrated, TEACHER_PROB_FLOOR, TEACHER_PROB_CEIL)
            source = "calibrated"
    chosen = expand_teacher_conviction(chosen, scale=TEACHER_EXPAND_SCALE)
    source = f"{source}+expand"
    return chosen.astype(np.float32), source


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
    chosen, _source = _calibrate_teacher_array(raw, calibrator)
    for idx, end in enumerate(ends):
        probs[end] = float(chosen[idx])
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
        active = probs[int(lookback) - 1 :] if int(lookback) > 0 else probs
        try:
            path_rel = str(Path(path).resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")
        except ValueError:
            path_rel = Path(path).name
        call_thr = float(params.get("confidence_call_threshold", 0.53))
        put_thr = float(params.get("confidence_put_threshold", 0.47))
        gray_mask = (active >= put_thr) & (active <= call_thr)
        logger.info(
            "[META] teacher %s n=%d lb=%d p=%.2f/%.2f/%.2f gray=%.0f%% ckpt=%s",
            bundle.symbol,
            int(probs.size),
            int(lookback),
            float(np.min(active)),
            float(np.mean(active)),
            float(np.max(active)),
            float(100.0 * np.mean(gray_mask)),
            path_rel,
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
