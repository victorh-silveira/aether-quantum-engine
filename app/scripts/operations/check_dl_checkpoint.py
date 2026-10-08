"""Confere integridade e geometria dos checkpoints TCN apos o treino."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


_APP = Path(__file__).resolve().parents[2]
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

from aether_paths import REPO_ROOT
from src.application.services.deep_learning.dl_features import FEATURE_DIM
from src.application.services.deep_learning.dl_model_checkpoint import load_model_checkpoint
from src.presentation.terminal.logger import setup_logger

_LOGGER = setup_logger("AETH.train", log_file=None)


def _load_settings() -> dict:
    """Carrega a configuracao operacional."""
    path = REPO_ROOT / "config" / "settings.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _expected_geometry(settings: dict) -> dict[str, int | str]:
    """Resolve geometria esperada do checkpoint a partir do SSOT."""
    dl = settings["deep_learning"]
    data = settings["data_handler"]
    timeframe = str(dl.get("train_timeframe", "micro")).strip().lower()
    granularity = (
        data["micro_granularity"] if timeframe in {"micro", "m5", "cycle", "settlement"} else data["granularity"]
    )
    return {
        "lookback": int(dl["lookback"]),
        "granularity": int(granularity),
        "label_horizon_bars": int(dl["label_horizon_bars"]),
        "label_mode": str(dl["label_mode"]),
        "feature_dim": FEATURE_DIM,
    }


def _checkpoint_paths(settings: dict, symbols: list[str]) -> list[Path]:
    """Resolve um checkpoint por simbolo configurado."""
    template = str(settings["deep_learning"].get("model_path_template", "data/dl/{symbol}.pth"))
    paths = []
    for symbol in symbols:
        raw = Path(template.format(symbol=symbol))
        paths.append(raw if raw.is_absolute() else REPO_ROOT / raw)
    return paths


def evaluate_checkpoint(path: Path, *, settings: dict) -> tuple[bool, str]:
    """Rejeita checkpoint ausente, corrompido ou com geometria incompativel."""
    if not path.is_file():
        return False, f"checkpoint ausente: {path}"
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        return False, f"{path.name}: checkpoint ilegivel ({exc})"
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict) or not payload["state_dict"]:
        return False, f"{path.name}: payload sem state_dict"
    for key, expected in _expected_geometry(settings).items():
        actual = payload.get(key)
        if actual is None or str(actual) != str(expected):
            return False, f"{path.name}: {key}={actual} != settings={expected}"
    if "norm_mean" not in payload or "norm_std" not in payload:
        return False, f"{path.name}: normalizacao ausente"
    if len(payload["norm_mean"]) != FEATURE_DIM or len(payload["norm_std"]) != FEATURE_DIM:
        return False, f"{path.name}: normalizacao incompativel"
    if load_model_checkpoint(path, params=settings["deep_learning"]) is None:
        return False, f"{path.name}: pesos incompativeis"
    return True, f"{path.name}: checkpoint tecnico compativel"


def main() -> int:
    """Valida todos os checkpoints exigidos pelo universo atual."""
    parser = argparse.ArgumentParser(description="Confere checkpoints TCN apos o treino.")
    parser.add_argument("--symbols", nargs="+", default=None)
    args = parser.parse_args()
    settings = _load_settings()
    symbols = args.symbols or settings.get("symbols") or ["1HZ75V"]
    results = [evaluate_checkpoint(path, settings=settings) for path in _checkpoint_paths(settings, symbols)]
    for _, message in results:
        _LOGGER.info("DL checkpoint | %s", message)
    return 0 if all(ok for ok, _ in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
