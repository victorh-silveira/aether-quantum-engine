from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import joblib


def atomic_joblib_dump(payload: Any, path: Path) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    joblib.dump(payload, tmp)
    os.replace(tmp, dest)
    return dest


def atomic_pickle_dump(payload: Any, path: Path) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(payload, handle)
    os.replace(tmp, dest)
    return dest
