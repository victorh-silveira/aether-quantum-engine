import pytest

from src.application.services.deep_learning.dl_features import FEATURE_DIM
from src.infrastructure.storage.torchscript_sanity import verify_torchscript_artifact_light_async


@pytest.mark.asyncio
async def test_verify_torchscript_artifact_light_async(tmp_path):
    path = tmp_path / "light_ts.pt"
    path.write_bytes(b"ts")
    manifest = {"feature_dim": FEATURE_DIM, "lookback": 48}
    await verify_torchscript_artifact_light_async(
        path,
        lookback=48,
        feature_dim=FEATURE_DIM,
        manifest=manifest,
    )


@pytest.mark.asyncio
async def test_verify_torchscript_artifact_light_missing_file(tmp_path):
    with pytest.raises(RuntimeError, match="ausente"):
        await verify_torchscript_artifact_light_async(
            tmp_path / "missing.pt",
            lookback=48,
            feature_dim=FEATURE_DIM,
        )
