"""Auditoria offline de nitidez (sharpness) do checkpoint TCN."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


_app_path = str(_repo_root() / "app")
if _app_path not in sys.path:
    sys.path.insert(0, _app_path)

import torch  # noqa: E402

from src.application.services.deep_learning.dl_calibration import (  # noqa: E402
    apply_calibrator_stable,
    calibrator_from_dict,
)
from src.application.services.deep_learning.dl_sharpness import (  # noqa: E402
    mean_sharpness,
    resolve_calibration_sharpness_cfg,
    sharpness_pass_fraction,
)


def _load_settings(repo: Path) -> dict:
    path = repo / "config" / "settings.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _hist_bins(probs: list[float], *, bins: int = 10) -> list[tuple[float, float, int]]:
    counts = [0 for _ in range(bins)]
    for prob in probs:
        p = min(max(float(prob), 0.0), 1.0)
        idx = min(bins - 1, int(p * bins))
        counts[idx] += 1
    out: list[tuple[float, float, int]] = []
    for i, count in enumerate(counts):
        lo = i / bins
        hi = (i + 1) / bins
        out.append((lo, hi, count))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Auditoria de sharpness do TCN")
    parser.add_argument("--symbol", default="OTC_SPC")
    parser.add_argument("--model-path", default="")
    parser.add_argument("--floor", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=512)
    args = parser.parse_args(argv)

    repo = _repo_root()
    settings = _load_settings(repo)
    dl = settings.get("deep_learning") if isinstance(settings, dict) else {}
    calib_cfg = dl.get("calibration") if isinstance(dl, dict) else {}
    sharpness_cfg = resolve_calibration_sharpness_cfg(calib_cfg if isinstance(calib_cfg, dict) else None)
    floor = float(args.floor) if float(args.floor) > 0.0 else float(sharpness_cfg["min_oos_sharpness"])

    template = str((dl or {}).get("model_path_template", "data/dl/{symbol}.pth"))
    model_path = Path(args.model_path) if args.model_path else repo / template.format(symbol=args.symbol)
    if not model_path.is_file():
        print(f"CHECKPOINT_AUSENTE path={model_path}")
        return 2

    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        print(f"CHECKPOINT_INVALIDO path={model_path}")
        return 2
    calibrator = calibrator_from_dict(
        payload.get("calibrator") if isinstance(payload.get("calibrator"), dict) else None
    )

    print(f"CHECKPOINT path={model_path}")
    print(f"FLOOR min_oos_sharpness={floor:.4f}")
    print(
        "META val_acc={va} brier={br} deploy_ok={ok} method={m} T={t:.3f}".format(
            va=payload.get("val_accuracy"),
            br=payload.get("val_brier"),
            ok=payload.get("deploy_ok"),
            m=calibrator.method,
            t=float(calibrator.temperature),
        )
    )

    synthetic = [0.5 + ((i % 21) - 10) * 0.005 for i in range(max(32, int(args.limit)))]
    calibrated = [float(apply_calibrator_stable(p, calibrator)) for p in synthetic]
    raw_sharp = mean_sharpness(synthetic)
    cal_sharp = mean_sharpness(calibrated)
    raw_pass = sharpness_pass_fraction(synthetic, floor=floor)
    cal_pass = sharpness_pass_fraction(calibrated, floor=floor)
    print(f"PROBE_SYNTH n={len(synthetic)} raw_sharp={raw_sharp:.4f} cal_sharp={cal_sharp:.4f}")
    print(f"PASS_FRAC floor={floor:.4f} raw={raw_pass:.3f} cal={cal_pass:.3f}")
    print("HIST_CAL bins:")
    for lo, hi, count in _hist_bins(calibrated):
        print(f"  [{lo:.1f},{hi:.1f}) count={count}")
    if cal_sharp + 1e-12 < floor:
        print("VEREDICTO calibrador reduz sharpness abaixo do piso quality-first")
        return 1
    print("VEREDICTO calibrador preserva sharpness no probe sintetico")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
