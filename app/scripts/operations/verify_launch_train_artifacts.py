"""Smoke pos-launch-train: checkpoint TCN + health loss/meta."""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import torch

_APP = Path(__file__).resolve().parents[2]
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

from aether_paths import REPO_ROOT


def main() -> int:
    ckpt_path = REPO_ROOT / "data" / "dl" / "1HZ75V.pth"
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print("val_accuracy", ck.get("val_accuracy"))
    print("deploy_ok", ck.get("deploy_ok"))
    print("lookback", ck.get("lookback"))
    print("granularity", ck.get("granularity"))
    print("label_horizon_bars", ck.get("label_horizon_bars"))
    print("feature_dim", ck.get("feature_dim"))
    print("pred_call_frac", ck.get("pred_call_frac"))
    print("minority_recall", ck.get("minority_recall"))
    with urllib.request.urlopen("http://127.0.0.1:8006/health", timeout=5) as r:
        health = json.load(r)
    print(
        "loss_buffer",
        health.get("buffer_n"),
        "bootstrap",
        health.get("bootstrap"),
        "auto",
        health.get("auto_learn_applied"),
    )
    with urllib.request.urlopen("http://127.0.0.1:8005/version", timeout=5) as r:
        meta = json.load(r)
    print("meta_version", meta.get("model_version"), "ready", meta.get("ready"))
    ok = bool(ck.get("deploy_ok")) and float(ck.get("val_accuracy") or 0.0) >= 0.55
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
