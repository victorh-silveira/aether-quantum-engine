"""Sweep offline de treino TCN por simbolo+timeframe (artefactos isolados + leaderboard)."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any


_APP = Path(__file__).resolve().parents[2]
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

import torch

from aether_paths import REPO_ROOT
from src.application.services.deep_learning.horizon_sweep import (
    build_horizon_candidates,
    load_horizon_sweep_knobs,
)
from src.application.services.deep_learning.tf_sweep_config import (
    candidate_artifact_dir,
    load_tf_sweep_knobs,
    resolve_repo_path,
)
from src.application.services.deep_learning.tf_sweep_promote import (
    patch_settings_for_sweep_train,
    write_json,
)
from src.application.services.deep_learning.tf_sweep_score import enrich_leaderboard_row
from src.application.services.deep_learning.tf_sweep_symbols import (
    normalize_sweep_symbol,
    resolve_sweep_symbols,
)
from src.domain.config_knobs import load_settings_json
from src.presentation.terminal.logger import setup_logger


TrainFn = Callable[[dict[str, Any], dict[str, Any], Path, str], dict[str, Any]]


def _read_checkpoint_metrics(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "deploy_ok": False,
            "val_accuracy": 0.0,
            "val_brier": 1.0,
            "oos_sharpness": 0.0,
            "error": f"checkpoint ausente: {path}",
        }
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        return {"deploy_ok": False, "val_accuracy": 0.0, "val_brier": 1.0, "error": "payload invalido"}
    return {
        "deploy_ok": bool(payload.get("deploy_ok", False)),
        "val_accuracy": float(payload.get("val_accuracy", 0.0) or 0.0),
        "val_brier": float(payload.get("val_brier", 1.0) or 1.0),
        "oos_sharpness": float(payload.get("oos_sharpness") or payload.get("raw_sharpness") or 0.0),
        "deploy_win_rate": float(payload.get("deploy_win_rate") or 0.0),
        "settle_wr": float(
            payload.get("deploy_settlement_win_rate")
            if payload.get("deploy_settlement_win_rate") is not None
            else (payload.get("deploy_win_rate") or 0.0)
        ),
        "settle_n": int(payload.get("deploy_settlement_n") or 0),
        "lookback": int(payload.get("lookback") or 0),
        "granularity_ckpt": int(payload.get("granularity") or 0),
        "error": None,
    }


def _swap_settings(settings_path: Path, patched: dict[str, Any]) -> Path:
    backup = settings_path.with_suffix(".json.tf_sweep.bak")
    if not backup.is_file():
        shutil.copy2(settings_path, backup)
    write_json(settings_path, patched)
    return backup


def _restore_settings(settings_path: Path, backup: Path) -> None:
    if backup.is_file():
        shutil.copy2(backup, settings_path)
        backup.unlink(missing_ok=True)


def default_train_candidate(
    patched_settings: dict[str, Any],
    candidate: dict[str, Any],
    artifact_dir: Path,
    symbol: str,
) -> dict[str, Any]:
    """Treina um simbolo+TF via train.py com settings temporariamente patchados."""
    _ = patched_settings
    artifact_dir.mkdir(parents=True, exist_ok=True)
    settings_path = REPO_ROOT / "config" / "settings.json"
    backup = _swap_settings(settings_path, patched_settings)
    sym = normalize_sweep_symbol(symbol)
    try:
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "app" / "train.py")],
            cwd=str(REPO_ROOT / "app"),
            check=False,
        )
        pth = artifact_dir / f"{sym}.pth"
        metrics = _read_checkpoint_metrics(pth)
        metrics["train_exit_code"] = int(proc.returncode)
        if proc.returncode != 0 and metrics.get("error") is None:
            metrics["error"] = f"train.py exit={proc.returncode}"
        metrics["tf"] = candidate["tf"]
        metrics["symbol"] = sym
        return metrics
    finally:
        _restore_settings(settings_path, backup)


def build_row_from_metrics(
    candidate: dict[str, Any],
    metrics: dict[str, Any],
    *,
    knobs: dict[str, Any],
    artifact_dir: Path,
    symbol: str,
) -> dict[str, Any]:
    """Monta linha do leaderboard a partir de metricas de ckpt/meta."""
    raw = {
        "symbol": normalize_sweep_symbol(symbol),
        "tf": candidate["tf"],
        "granularity": candidate["micro_granularity"],
        "micro_granularity": candidate["micro_granularity"],
        "macro_granularity": candidate["macro_granularity"],
        "mini_granularity": candidate["mini_granularity"],
        "duration": candidate["duration"],
        "duration_unit": candidate["duration_unit"],
        "lookback": candidate["lookback"],
        "history_bars": candidate["history_bars"],
        "label_horizon_bars": candidate["label_horizon_bars"],
        "deploy_ok": bool(metrics.get("deploy_ok")),
        "val_accuracy": float(metrics.get("val_accuracy") or 0.0),
        "val_brier": float(metrics.get("val_brier") or 1.0),
        "oos_sharpness": float(metrics.get("oos_sharpness") or 0.0),
        "mini_win_rate": float(metrics.get("deploy_win_rate") or 0.0),
        "settle_wr": float(
            metrics["settle_wr"] if metrics.get("settle_wr") is not None else (metrics.get("deploy_win_rate") or 0.0)
        ),
        "settle_n": int(metrics.get("settle_n") or 0),
        "meta_ir": float(metrics.get("meta_ir") or 0.0),
        "meta_payoff_zscore": float(metrics.get("meta_payoff_zscore") or 0.0),
        "artifact_dir": str(artifact_dir),
        "error": metrics.get("error"),
        "train_exit_code": metrics.get("train_exit_code"),
    }
    return enrich_leaderboard_row(raw, knobs=knobs)


def run_tf_sweep(
    *,
    settings: dict[str, Any] | None = None,
    knobs: dict[str, Any] | None = None,
    candidates: list[dict[str, Any]] | None = None,
    train_fn: TrainFn | None = None,
    only_tf: set[str] | None = None,
    only_symbols: set[str] | None = None,
    dry_run: bool = False,
    repo_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Executa o sweep simbolo×TF e devolve leaderboard enriquecido."""
    root = repo_root if repo_root is not None else REPO_ROOT
    full = settings if isinstance(settings, dict) else load_settings_json()
    sweep_knobs = knobs if isinstance(knobs, dict) else load_tf_sweep_knobs(full)
    rows_in = candidates if candidates is not None else build_horizon_candidates(full)
    if only_tf:
        wanted = {t.strip().upper() for t in only_tf}
        rows_in = [c for c in rows_in if c["tf"] in wanted]
    symbols = resolve_sweep_symbols(sweep_knobs)
    if only_symbols:
        wanted_sym = {normalize_sweep_symbol(s) for s in only_symbols}
        symbols = [s for s in symbols if s in wanted_sym]
    trainer = train_fn or default_train_candidate
    board: list[dict[str, Any]] = []
    for symbol in symbols:
        for candidate in rows_in:
            art = candidate_artifact_dir(
                str(sweep_knobs["artifact_root"]),
                candidate["tf"],
                symbol=symbol,
                repo_root=root,
            )
            patched = patch_settings_for_sweep_train(
                full,
                candidate,
                artifact_root=str(sweep_knobs["artifact_root"]),
                symbol=symbol,
                train_deploy_retries=int(sweep_knobs.get("train_deploy_retries", 1)),
                disable_infra=bool(sweep_knobs.get("disable_infra_during_sweep", True)),
            )
            if dry_run:
                metrics = {
                    "deploy_ok": False,
                    "val_accuracy": 0.0,
                    "val_brier": 1.0,
                    "oos_sharpness": 0.0,
                    "error": "dry_run",
                    "train_exit_code": 0,
                    "symbol": symbol,
                }
                write_json(art / "settings_overlay.json", patched)
            else:
                metrics = trainer(patched, candidate, art, symbol)
            board.append(
                build_row_from_metrics(
                    candidate,
                    metrics,
                    knobs=sweep_knobs,
                    artifact_dir=art,
                    symbol=symbol,
                )
            )
    leaderboard_path = resolve_repo_path(str(sweep_knobs["leaderboard_path"]), repo_root=root)
    write_json(
        leaderboard_path,
        {
            "version": 1,
            "payout_for_breakeven": sweep_knobs["payout_for_breakeven"],
            "min_edge_vs_breakeven": sweep_knobs["min_edge_vs_breakeven"],
            "symbols": symbols,
            "rows": board,
        },
    )
    return board


def main(argv: list[str] | None = None) -> int:
    """CLI do sweep de horizonte H1-H5 (offline)."""
    parser = argparse.ArgumentParser(description="Sweep de treino de horizonte H1-H5 (offline)")
    parser.add_argument("--dry-run", action="store_true", help="So grava overlays e leaderboard")
    parser.add_argument("--only", nargs="*", default=None, help="Filtra celulas (ex.: H1 H3)")
    parser.add_argument("--only-symbol", nargs="*", default=None, help="Filtra simbolos (ex.: R_10)")
    parser.add_argument("--from-checkpoints", action="store_true", help="So le ckpts ja existentes")
    args = parser.parse_args(argv)
    log = setup_logger("HORIZON")
    settings = load_settings_json()
    knobs = load_horizon_sweep_knobs(settings)
    if not knobs.get("enabled", True):
        log.error("horizon_sweep.enabled=false")
        return 2
    only = {t.strip().upper() for t in args.only} if args.only else None
    only_sym = set(args.only_symbol) if args.only_symbol else None

    def _from_ckpt(
        patched: dict[str, Any],
        candidate: dict[str, Any],
        art: Path,
        symbol: str,
    ) -> dict[str, Any]:
        _ = (patched, candidate)
        metrics = _read_checkpoint_metrics(art / f"{normalize_sweep_symbol(symbol)}.pth")
        metrics["symbol"] = normalize_sweep_symbol(symbol)
        return metrics

    train_fn: TrainFn | None = _from_ckpt if args.from_checkpoints else None
    board = run_tf_sweep(
        knobs=knobs,
        candidates=build_horizon_candidates(settings, n_bars=knobs.get("n_bars")),
        only_tf=only,
        only_symbols=only_sym,
        dry_run=bool(args.dry_run),
        train_fn=train_fn,
    )
    eligible = [r for r in board if r.get("eligible")]
    log.info(
        "[HORIZON] candidatos=%s elegiveis=%s leaderboard=%s",
        len(board),
        len(eligible),
        knobs["leaderboard_path"],
    )
    for row in board:
        log.info(
            "[HORIZON] symbol=%s tf=%s settle=%.4f label_acc=%.4f edge_vs_be=%.4f deploy=%s eligible=%s score=%s err=%s",
            row.get("symbol"),
            row.get("tf"),
            float(row.get("rank_wr") or row.get("settle_wr") or 0.0),
            float(row.get("val_accuracy") or 0.0),
            float(row.get("edge_vs_be") or 0.0),
            row.get("deploy_ok"),
            row.get("eligible"),
            row.get("score"),
            row.get("error"),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
