"""Treino meta completo sem publicar candidatos de qualidade insuficiente."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import joblib
import numpy as np
import polars as pl
import pytest

from scripts.operations import train_meta_classifier as meta


@pytest.mark.asyncio
@pytest.mark.parametrize("qualified", [False, True])
async def test_meta_candidate_stays_outside_live_model_directory(tmp_path: Path, monkeypatch, *, qualified: bool):
    candidate = tmp_path / "diagnostics" / "meta_candidate.joblib"
    live = tmp_path / "live" / "meta_lgbm.pkl"
    monkeypatch.setattr(meta, "DEFAULT_CANDIDATE_OUTPUT", candidate)
    scores = {
        "feature_dim": 23,
        "model_type": "regressor",
        "oos_payoff_zscore_mean": 0.10 if qualified else -0.10,
        "oos_information_ratio": 0.20 if qualified else -0.20,
    }
    frame = pl.DataFrame({"micro_price_velocity": [0.0, 0.2, -0.1]})
    y = np.array([1.0, -1.0, 0.5], dtype=np.float32)
    settings = {"deep_learning": {"lookback": 30}, "data_handler": {"micro_granularity": 300}}
    with (
        patch.object(meta, "resolve_training_bundles", new_callable=AsyncMock, return_value=[object()]),
        patch.object(meta, "assert_bundles_match_granularity"),
        patch.object(meta, "_teacher_probs", return_value={}),
        patch.object(meta, "build_paired_training_dataset", return_value=(frame, y, np.array([0.5] * 3), None, {})),
        patch.object(meta, "run_optuna_study", return_value=(object(), scores, 0.10, 0.12)) as study,
        patch.object(
            meta, "build_training_summary", side_effect=lambda **kwargs: {"output": str(kwargs["output_path"])}
        ),
    ):
        summary = await meta.train_meta_classifier(
            settings=settings,
            dsn="unused",
            symbols=["1HZ75V"],
            granularity=300,
            bars=5000,
            trials=2,
            output_path=live,
            source="auto",
            candidate_on_low_quality=True,
        )
    exported = live if qualified else candidate
    assert Path(summary["output"]) == exported
    assert exported.is_file()
    assert joblib.load(exported)["deploy_qualified"] is qualified
    assert not (candidate if qualified else live).exists()
    assert study.call_args.kwargs["allow_unqualified"] is True


@pytest.mark.asyncio
async def test_meta_strict_mode_still_rejects_bad_oos(tmp_path: Path):
    live = tmp_path / "meta_lgbm.pkl"
    frame = pl.DataFrame({"micro_price_velocity": [0.0, 0.2, -0.1]})
    y = np.array([1.0, -1.0, 0.5], dtype=np.float32)
    with (
        patch.object(meta, "resolve_training_bundles", new_callable=AsyncMock, return_value=[object()]),
        patch.object(meta, "assert_bundles_match_granularity"),
        patch.object(meta, "_teacher_probs", return_value={}),
        patch.object(meta, "build_paired_training_dataset", return_value=(frame, y, np.array([0.5] * 3), None, {})),
        patch.object(meta, "run_optuna_study", return_value=(object(), {"oos_payoff_zscore_mean": -1.0}, 0.10, 0.12)),
        pytest.raises(RuntimeError, match="Export meta bloqueado"),
    ):
        await meta.train_meta_classifier(
            settings={"deep_learning": {"lookback": 30}},
            dsn="unused",
            symbols=["1HZ75V"],
            granularity=300,
            bars=5000,
            trials=2,
            output_path=live,
            source="auto",
        )
    assert not live.exists()
