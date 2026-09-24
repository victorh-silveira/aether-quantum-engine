"""Valida checkpoint DL apos treino: deploy_ok, ACC/settle e geometria."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import torch


_APP = Path(__file__).resolve().parents[2]
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

from aether_paths import REPO_ROOT
from src.application.services.deep_learning.dl_gate_config import parse_deploy_gate_config
from src.presentation.terminal.logger import setup_logger

_LOGGER = setup_logger("AETH.train", log_file=None)


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


def _expected_geometry(settings: dict) -> tuple[int | None, int | None, int | None]:
    dl = settings.get("deep_learning") if isinstance(settings.get("deep_learning"), dict) else {}
    data = settings.get("data_handler") if isinstance(settings.get("data_handler"), dict) else {}
    if not isinstance(dl, dict):
        return None, None, None
    lookback = int(dl["lookback"]) if "lookback" in dl else None
    horizon = int(dl["label_horizon_bars"]) if "label_horizon_bars" in dl else None
    tf = str(dl.get("train_timeframe", "micro")).strip().lower()
    if not isinstance(data, dict):
        return lookback, None, horizon
    if tf in ("micro", "m5", "cycle", "settlement"):
        gran = int(data["micro_granularity"]) if "micro_granularity" in data else None
    else:
        gran = int(data["granularity"]) if "granularity" in data else None
    return lookback, gran, horizon


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


def evaluate_checkpoint(
    path: Path,
    *,
    soft_min: float,
    settings: dict | None = None,
    meta_path: Path | None = None,
) -> tuple[bool, str]:
    if not path.is_file():
        return False, f"checkpoint ausente: {path}"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        return False, f"payload invalido: {path}"
    if isinstance(settings, dict):
        exp_lb, exp_gran, exp_h = _expected_geometry(settings)
        got_lb = payload.get("lookback")
        got_gran = payload.get("granularity")
        got_h = payload.get("label_horizon_bars")
        if exp_lb is not None and got_lb is not None and int(got_lb) != int(exp_lb):
            return False, f"{path.name}: lookback={got_lb} != settings={exp_lb} (treino incompleto / ckpt antigo)"
        if exp_gran is not None and got_gran is not None and int(got_gran) != int(exp_gran):
            return False, (
                f"{path.name}: granularity={got_gran} != settings={exp_gran} (treino incompleto / ckpt antigo)"
            )
        if exp_h is not None and (got_h is None or int(got_h) != int(exp_h)):
            return False, (
                f"{path.name}: label_horizon_bars={got_h} != settings={exp_h} (treino incompleto / ckpt antigo)"
            )
        dl_settings = settings.get("deep_learning")
        expected_mode = dl_settings.get("label_mode") if isinstance(dl_settings, dict) else None
        if expected_mode is not None and payload.get("label_mode") != expected_mode:
            return False, f"{path.name}: label_mode={payload.get('label_mode')} != settings={expected_mode}"
    dl = {}
    if isinstance(settings, dict) and isinstance(settings.get("deep_learning"), dict):
        dl = settings["deep_learning"]
    gate_cfg = parse_deploy_gate_config(dl)
    val_acc = float(payload.get("val_accuracy", payload.get("val_acc", 0.0)) or 0.0)
    val_brier = float(payload.get("val_brier", 1.0) or 1.0)
    stored_ok = bool(payload.get("deploy_ok", False))
    provisional_ok = bool(payload.get("deploy_provisional_ok", False))
    settlement_lcb = payload.get("deploy_settlement_wilson_lcb")
    if val_acc + 1e-9 < soft_min:
        return False, f"{path.name}: val_acc={val_acc:.4f} < soft_min={soft_min:.4f}"
    label_call = payload.get("label_call_frac")
    pred_call = payload.get("pred_call_frac")
    minority_rec = payload.get("minority_recall")
    if bool(gate_cfg.get("reject_majority_collapse", False)) and (
        label_call is None or pred_call is None or minority_rec is None
    ):
        return False, (
            f"{path.name}: telemetria de collapse ausente "
            "(label_call_frac/pred_call_frac/minority_recall) — retreine com gate atual"
        )
    if meta_path is not None and meta_path.is_file():
        try:
            meta_bundle = joblib.load(meta_path)
            if isinstance(meta_bundle, dict) and bool(meta_bundle.get("deploy_qualified", False)):
                ir = float(meta_bundle.get("oos_information_ratio", 0.0) or 0.0)
                z = float(meta_bundle.get("oos_payoff_zscore_mean", 0.0) or 0.0)
                if ir >= 0.50 and z >= 0.010:
                    return True, (
                        f"{path.name}: Two-Stage Stacking qualificado "
                        f"(TCN val_acc={val_acc:.4f} + Meta IR={ir:.2f} Z={z:.3f})"
                    )
        except Exception as exc:
            _LOGGER.debug("Falha ao avaliar meta_path para deploy gate: %s", exc)
    if not stored_ok:
        settlement_wr = float(payload.get("deploy_settlement_win_rate", 0.0) or 0.0)
        settlement_n = int(payload.get("deploy_settlement_n", 0) or 0)
        settlement_brier = float(payload.get("deploy_settlement_brier", 1.0) or 1.0)
        if (
            provisional_ok
            and payload.get("deploy_settlement_source") == "broker_tick_audit"
            and (
                settlement_n >= int(gate_cfg["provisional_min_trades"])
                and settlement_wr + 1e-9 >= float(gate_cfg["provisional_min_win_rate"])
                and settlement_brier + 1e-9 < float(gate_cfg["provisional_max_brier"])
            )
        ):
            return True, (
                f"{path.name}: deploy_provisional=true "
                f"(settle_wr={settlement_wr:.4f} n={settlement_n} brier={settlement_brier:.4f})"
            )
        return False, (
            f"{path.name}: deploy_ok=false "
            f"(val_acc={val_acc:.4f} val_brier={val_brier:.4f} "
            f"pred_call={pred_call} minority_rec={minority_rec} "
            "sem evidencia OOS de settlement qualificada)"
        )
    if (
        bool(gate_cfg.get("require_broker_settlement", False))
        and payload.get("deploy_settlement_source") != "broker_tick_audit"
    ):
        return False, f"{path.name}: settlement M5 e apenas proxy; evidencias broker/tick auditadas ausentes"
    if settlement_lcb is None or float(settlement_lcb) + 1e-9 < float(gate_cfg["min_win_rate"]):
        return False, (
            f"{path.name}: checkpoint sem evidencia Wilson de settlement suficiente "
            f"(lcb={settlement_lcb}; min_win_rate={float(gate_cfg['min_win_rate']):.6f})"
        )
    return True, f"{path.name}: deploy_ok=true val_acc={val_acc:.4f}"


def main() -> int:
    settings = _load_settings()
    logger = _LOGGER
    parser = argparse.ArgumentParser(description="Gate ACC/deploy apos treino DL.")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--soft-min", type=float, default=None)
    parser.add_argument("--with-meta", action="store_true", default=False)
    args = parser.parse_args()
    soft_min = float(args.soft_min) if args.soft_min is not None else _soft_min_acc(settings)
    raw_symbols = args.symbols if args.symbols is not None else settings.get("symbols") or ["1HZ75V"]
    symbols = [str(s) for s in raw_symbols]
    meta_path = REPO_ROOT / "infra" / "docker" / "meta-models" / "meta_lgbm.pkl" if args.with_meta else None
    ok_all = True
    for path in _checkpoint_paths(settings, symbols):
        ok, msg = evaluate_checkpoint(path, soft_min=soft_min, settings=settings, meta_path=meta_path)
        logger.info("DL gate | %s", msg)
        ok_all = ok_all and ok
    if not ok_all:
        logger.warning(
            "DL gate de qualificacao OOS reprovado: ACC/Brier/settle/geometria. "
            "Treino meta pode continuar; somente checkpoint local tecnicamente compativel pode operar "
            "com teto de stake em DEMO e REAL."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
