"""Promocao atomica do vencedor do sweep de horizonte para o SSOT."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import torch

from src.application.services.deep_learning.horizon_sweep import load_horizon_sweep_knobs
from src.application.services.deep_learning.tf_sweep_config import (
    candidate_artifact_dir,
    resolve_repo_path,
)
from src.application.services.deep_learning.tf_sweep_score import pick_tf_winner
from src.application.services.deep_learning.tf_sweep_symbols import (
    clear_other_live_checkpoints,
    normalize_sweep_symbol,
    patch_settings_for_symbol,
    write_trading_symbols_module,
)


def patch_settings_for_candidate(
    settings: dict[str, Any],
    candidate: dict[str, Any],
    *,
    ops_contract_duration_minutes: int | None = None,
) -> dict[str, Any]:
    """Retorna copia de settings com ciclo/horizon da celula (sem reescalar lookback).

    Treino (ops=None): params.duration = duration da celula.
    Promote (ops=int): params.duration fixo em ops; label_horizon_bars vem do winner.
    """
    out = copy.deepcopy(settings)
    micro = int(candidate["micro_granularity"])
    macro = int(candidate["macro_granularity"])
    mini = int(candidate.get("mini_granularity") or micro)
    if ops_contract_duration_minutes is None:
        duration = int(candidate["duration"])
        duration_unit = str(candidate.get("duration_unit") or "m")
    else:
        duration = max(1, int(ops_contract_duration_minutes))
        duration_unit = "m"
    label_h = int(candidate.get("label_horizon_bars") or 1)
    data = out.setdefault("data_handler", {})
    if not isinstance(data, dict):
        raise ValueError("data_handler invalido")
    data["micro_granularity"] = micro
    data["mini_granularity"] = mini
    data["granularity"] = macro
    dl = out.setdefault("deep_learning", {})
    if not isinstance(dl, dict):
        raise ValueError("deep_learning invalido")
    dl["train_timeframe"] = str(candidate.get("train_timeframe") or "micro")
    dl["label_horizon_bars"] = label_h
    risk = out.setdefault("risk_management", {})
    if not isinstance(risk, dict):
        raise ValueError("risk_management invalido")
    params = risk.setdefault("params", {})
    if not isinstance(params, dict):
        raise ValueError("risk_management.params invalido")
    params["duration"] = duration
    params["duration_unit"] = duration_unit
    kelly = risk.setdefault("kelly", {})
    if isinstance(kelly, dict):
        kelly["cycle_stake_baseline_seconds"] = micro
    orch = out.setdefault("orchestrator", {})
    if not isinstance(orch, dict):
        raise ValueError("orchestrator invalido")
    orch["cycle_interval_seconds"] = micro
    orch["signature_boundary_seconds"] = micro
    orch["exec_empty_retry_seconds"] = micro
    if candidate.get("lookback") is not None:
        dl["lookback"] = int(candidate["lookback"])
    if candidate.get("history_bars") is not None:
        dl["training_history_bars"] = int(candidate["history_bars"])
    return out


def patch_settings_for_sweep_train(
    settings: dict[str, Any],
    candidate: dict[str, Any],
    *,
    artifact_root: str,
    symbol: str = "1HZ75V",
    train_deploy_retries: int = 1,
    disable_infra: bool = True,
    quiet_train_logs: bool | None = None,
) -> dict[str, Any]:
    """Settings de treino isolado: path de ckpt em data/dl/sweep/{symbol}/{tf}."""
    out = patch_settings_for_candidate(settings, candidate)
    patch_settings_for_symbol(out, symbol)
    dl = out["deep_learning"]
    tf = str(candidate["tf"]).strip().upper()
    sym = normalize_sweep_symbol(symbol)
    dl["model_path_template"] = f"{artifact_root.rstrip('/')}/{sym}/{tf}/{{symbol}}.pth"
    dl["train_deploy_retries"] = max(1, int(train_deploy_retries))
    if disable_infra:
        infra = out.setdefault("infra", {})
        if isinstance(infra, dict):
            infra["enabled"] = False
    quiet = quiet_train_logs
    if quiet is None:
        h_src = settings.get("deep_learning") if isinstance(settings.get("deep_learning"), dict) else {}
        h_block = h_src.get("horizon_sweep") if isinstance(h_src, dict) else {}
        quiet = bool(h_block.get("quiet_train_logs", True)) if isinstance(h_block, dict) else True
    if quiet:
        log_block = out.setdefault("logging", {})
        if isinstance(log_block, dict):
            log_block["level"] = "CRITICAL"
        dl["training_log_every_n_epochs"] = 10**9
    return out


def checkpoint_paths_for_tf(
    *,
    artifact_root: str,
    tf: str,
    symbol: str = "1HZ75V",
    repo_root: Path | None = None,
) -> tuple[Path, Path]:
    """Retorna (pth, torchscript) no diretorio do simbolo+TF."""
    folder = candidate_artifact_dir(artifact_root, tf, symbol=symbol, repo_root=repo_root)
    sym = normalize_sweep_symbol(symbol)
    return folder / f"{sym}.pth", folder / f"{sym}_ts.pt"


def _stamp_checkpoint_deploy_ok(path: Path) -> None:
    """Marca deploy_ok no ckpt promovido (elegibilidade do sweep e por ACC/edge)."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint invalido para stamp: {path}")
    if bool(payload.get("deploy_ok", False)):
        return
    payload["deploy_ok"] = True
    torch.save(payload, path)


def promote_artifacts(
    *,
    artifact_root: str,
    tf: str,
    symbol: str = "1HZ75V",
    dest_dir: Path | None = None,
    repo_root: Path | None = None,
) -> list[Path]:
    """Copia ckpt/TorchScript do sweep para data/dl live."""
    src_pth, src_ts = checkpoint_paths_for_tf(artifact_root=artifact_root, tf=tf, symbol=symbol, repo_root=repo_root)
    if not src_pth.is_file():
        raise FileNotFoundError(f"checkpoint sweep ausente: {src_pth}")
    dest = dest_dir if dest_dir is not None else resolve_repo_path("data/dl", repo_root=repo_root)
    dest.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    dest_pth = dest / f"{symbol}.pth"
    shutil.copy2(src_pth, dest_pth)
    _stamp_checkpoint_deploy_ok(dest_pth)
    copied.append(dest_pth)
    if src_ts.is_file():
        dest_ts = dest / f"{symbol}_ts.pt"
        shutil.copy2(src_ts, dest_ts)
        copied.append(dest_ts)
    return copied


def promote_winner_from_leaderboard(
    rows: list[dict[str, Any]],
    settings: dict[str, Any],
    *,
    artifact_root: str,
    symbol: str | None = None,
    dest_dir: Path | None = None,
    repo_root: Path | None = None,
    copy_artifacts: bool = True,
    write_symbols_module: bool = True,
    symbols_module_path: Path | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any], list[Path]]:
    """Escolhe vencedor, patcha settings/simbolo e opcionalmente copia artefactos.

    Retorna (winner_or_none, settings_out, copied_paths). Sem elegivel: settings intacto.
    """
    winner = pick_tf_winner(rows)
    if winner is None:
        return None, settings, []
    win_symbol = normalize_sweep_symbol(symbol or winner.get("symbol") or "1HZ75V")
    candidate = {
        "tf": winner["tf"],
        "micro_granularity": int(winner.get("granularity") or winner.get("micro_granularity")),
        "macro_granularity": int(winner["macro_granularity"]),
        "mini_granularity": int(winner.get("mini_granularity") or winner.get("granularity")),
        "duration": int(winner["duration"]),
        "duration_unit": str(winner.get("duration_unit") or "m"),
        "lookback": int(winner.get("lookback") or 720),
        "history_bars": int(winner.get("history_bars") or 2000),
        "label_horizon_bars": int(winner.get("label_horizon_bars") or 1),
        "train_timeframe": "micro",
    }
    ops_m = int(load_horizon_sweep_knobs(settings)["ops_contract_duration_minutes"])
    patched = patch_settings_for_candidate(settings, candidate, ops_contract_duration_minutes=ops_m)
    patch_settings_for_symbol(patched, win_symbol)
    dl = patched.setdefault("deep_learning", {})
    if isinstance(dl, dict):
        dl["model_path_template"] = "data/dl/{symbol}.pth"
    if write_symbols_module:
        write_trading_symbols_module(win_symbol, path=symbols_module_path)
    copied: list[Path] = []
    if copy_artifacts:
        dest = dest_dir if dest_dir is not None else resolve_repo_path("data/dl", repo_root=repo_root)
        copied = promote_artifacts(
            artifact_root=artifact_root,
            tf=str(winner["tf"]),
            symbol=win_symbol,
            dest_dir=dest,
            repo_root=repo_root,
        )
        clear_other_live_checkpoints(dest, win_symbol)
    return winner, patched, copied


def write_json(path: Path, payload: Any) -> None:
    """Grava JSON com indentacao estavel."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
