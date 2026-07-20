from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.operations.train_meta_data import OhlcBundle
from scripts.operations.train_meta_teacher import (
    infer_teacher_probs_for_bundle,
    infer_teacher_probs_from_checkpoints,
    load_teacher_probs_from_checkpoints,
)
from src.application.services.deep_learning.dl_feature_build import precompute_price_series
from src.application.services.deep_learning.dl_feature_matrix import build_feature_matrix
from src.application.services.deep_learning.dl_features import FEATURE_DIM
from src.application.services.deep_learning.dl_model_checkpoint import save_model_checkpoint
from src.application.services.deep_learning.dl_model_factory import create_direction_model
from src.application.services.deep_learning.model import fit_norm_stats


def _bundle(symbol: str, n: int = 120, gran: int = 120) -> OhlcBundle:
    closes = np.linspace(100.0, 110.0, n, dtype=np.float64) + np.sin(np.linspace(0.0, 8.0, n))
    high = closes + 0.2
    low = closes - 0.2
    open_ = closes.copy()
    epochs = np.arange(1_700_000_000, 1_700_000_000 + n * gran, gran, dtype=np.int64)
    return OhlcBundle(
        symbol=symbol,
        granularity=gran,
        closes=closes,
        open_=open_,
        high=high,
        low=low,
        epochs=epochs,
        source="test",
    )


def _norm_for_bundle(bundle: OhlcBundle, lookback: int):
    series = precompute_price_series(
        bundle.closes,
        granularity=bundle.granularity,
        symbol=bundle.symbol,
        open_=bundle.open_,
        high=bundle.high,
        low=bundle.low,
    )
    matrix = build_feature_matrix(series)
    seq = np.stack(
        [matrix[i - lookback + 1 : i + 1] for i in range(lookback - 1, lookback + 8)],
        axis=0,
    ).astype(np.float32)
    return fit_norm_stats(seq)


def test_infer_teacher_probs_for_bundle_length_and_warmup():
    lookback = 16
    bundle = _bundle("R_10", n=80)
    model = create_direction_model(arch="tcn", input_dim=FEATURE_DIM)
    model.eval()
    probs = infer_teacher_probs_for_bundle(
        bundle,
        model=model,
        norm_stats=_norm_for_bundle(bundle, lookback),
        lookback=lookback,
        batch_size=8,
    )
    assert probs.shape == (len(bundle.closes),)
    assert np.all(probs[: lookback - 1] == pytest.approx(0.5))
    assert np.all((probs >= 0.05) & (probs <= 0.95))


def test_infer_teacher_probs_from_checkpoints_loads_saved_model(tmp_path: Path):
    bundle = _bundle("R_10", n=96)
    assert (
        infer_teacher_probs_from_checkpoints(
            [bundle],
            model_path_template=str(tmp_path / "{symbol}.pth"),
        )
        == {}
    )
    lookback = 16
    model = create_direction_model(arch="tcn", input_dim=FEATURE_DIM)
    norm = _norm_for_bundle(bundle, lookback)
    path = tmp_path / "R_10.pth"
    save_model_checkpoint(
        path,
        model,
        norm,
        last_candle_epoch=int(bundle.epochs[-1]),
        lookback=lookback,
        val_accuracy=0.55,
        val_brier=0.24,
        deploy_ok=False,
    )
    loaded = infer_teacher_probs_from_checkpoints(
        [bundle],
        model_path_template=str(tmp_path / "{symbol}.pth"),
        dl_params={"arch": "tcn"},
        batch_size=16,
    )
    assert "R_10" in loaded
    assert loaded["R_10"].shape == (len(bundle.closes),)


def test_load_teacher_probs_from_checkpoints_embedded_arrays(tmp_path: Path):
    path = tmp_path / "R_10.pth"
    torch.save({"teacher_probs": np.linspace(0.2, 0.8, 40).astype(np.float32)}, path)
    loaded = load_teacher_probs_from_checkpoints(
        ["R_10"],
        model_path_template=str(tmp_path / "{symbol}.pth"),
        lookback=16,
        repo_root=tmp_path,
    )
    assert "R_10" in loaded
    assert loaded["R_10"].shape == (40,)
