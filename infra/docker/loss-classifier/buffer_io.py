from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


BUFFER_FILENAME = "learn_buffer.pkl"


def buffer_path(models_dir: Path) -> Path:
    return Path(models_dir) / BUFFER_FILENAME


def save_learn_buffer(models_dir: Path, buffer_x: list[list[float]], buffer_y: list[int]) -> Path:
    path = buffer_path(models_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"x": list(buffer_x), "y": list(buffer_y)}, path)
    return path


def load_learn_buffer(models_dir: Path) -> tuple[list[list[float]], list[int]] | None:
    path = buffer_path(models_dir)
    if not path.is_file():
        return None
    try:
        payload = joblib.load(path)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    raw_x = payload.get("x")
    raw_y = payload.get("y")
    if not isinstance(raw_x, list) or not isinstance(raw_y, list):
        return None
    if len(raw_x) != len(raw_y):
        return None
    out_x: list[list[float]] = []
    out_y: list[int] = []
    for row, label in zip(raw_x, raw_y, strict=True):
        if not isinstance(row, (list, tuple)):
            return None
        try:
            out_x.append([float(v) for v in row])
            out_y.append(int(label))
        except (TypeError, ValueError):
            return None
    return out_x, out_y


def buffer_class_counts(buffer_y: list[int]) -> dict[str, Any]:
    wins = sum(1 for y in buffer_y if int(y) == 0)
    losses = sum(1 for y in buffer_y if int(y) == 1)
    return {"win": int(wins), "loss": int(losses), "n": int(len(buffer_y))}
