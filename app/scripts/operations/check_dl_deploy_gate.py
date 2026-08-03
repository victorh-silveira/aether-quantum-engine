"""Valida checkpoint DL apos treino: deploy_ok e ACC >= soft_min (padrao 0.53)."""

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
from src.presentation.terminal.logger import setup_logger


def _load_settings() -> dict:
    path = REPO_ROOT / "config" / "settings.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _soft_min_acc(settings: dict) -> float:
    dl = settings.get("deep_learning") if isinstance(settings.get("deep_learning"), dict) else {}
    gate = dl.get("deploy_gate") if isinstance(dl, dict) and isinstance(dl.get("deploy_gate"), dict) else {}
    if isinstance(gate, dict) and gate.get("soft_min_val_accuracy") is not None:
        return float(gate["soft_min_val_accuracy"])
    return float(dl.get("min_val_accuracy", 0.53)) if isinstance(dl, dict) else 0.53


def _checkpoint_paths(settings: dict, symbols: list[str]) -> list[Path]:
    dl = settings.get("deep_learning") if isinstance(settings.get("deep_learning"), dict) else {}
    template = (
        str(dl.get("model_path_template", "data/dl/{symbol}.pth")) if isinstance(dl, dict) else "data/dl/{symbol}.pth"
    )
    out: list[Path] = []
    for symbol in symbols:
        raw = template.format(symbol=symbol)
        path = Path(raw) if Path(raw).is_absolute() else REPO_ROOT / raw
        out.append(path)
    return out


def evaluate_checkpoint(path: Path, *, soft_min: float) -> tuple[bool, str]:
    if not path.is_file():
        return False, f"checkpoint ausente: {path}"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        return False, f"payload invalido: {path}"
    val_acc = float(payload.get("val_accuracy", payload.get("val_acc", 0.0)) or 0.0)
    deploy_ok = bool(payload.get("deploy_ok", False))
    if val_acc + 1e-9 < soft_min:
        return False, f"{path.name}: val_acc={val_acc:.4f} < soft_min={soft_min:.4f}"
    if not deploy_ok:
        return False, f"{path.name}: deploy_ok=false (val_acc={val_acc:.4f})"
    return True, f"{path.name}: deploy_ok=true val_acc={val_acc:.4f}"


def main() -> int:
    settings = _load_settings()
    logger = setup_logger("AETH.train", log_file=None)
    parser = argparse.ArgumentParser(description="Gate ACC/deploy apos treino DL.")
    parser.add_argument("--symbols", nargs="+", default=["R_10"])
    parser.add_argument("--soft-min", type=float, default=None)
    args = parser.parse_args()
    soft_min = float(args.soft_min) if args.soft_min is not None else _soft_min_acc(settings)
    ok_all = True
    for path in _checkpoint_paths(settings, [str(s) for s in args.symbols]):
        ok, msg = evaluate_checkpoint(path, soft_min=soft_min)
        logger.info("DL gate | %s", msg)
        ok_all = ok_all and ok
    if not ok_all:
        logger.error("DL gate falhou: ACC<0.53 ou deploy_ok=false — meta abortado. Retreine.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
