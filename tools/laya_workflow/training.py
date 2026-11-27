"""Preparacao de tokens e treinamento CUDA com calibracao temporal externa."""

import json
from pathlib import Path
from types import SimpleNamespace

from src.infrastructure.inference.laya_context import encode_laya_context
from tools.laya_workflow.artifacts import read_json, read_rows, verify_dataset


def cuda_preflight(torch) -> dict:
    """Verifica GPU antes de baixar ou carregar pesos; CPU nao e fallback de treino."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA indisponivel: execute train em runtime PyTorch CUDA; use train --dry-run nesta maquina"
        )
    free, total = torch.cuda.mem_get_info()
    if free < 3 * 1024**3:
        raise RuntimeError("Treinamento exige pelo menos 3 GiB livres na GPU")
    return {"device": torch.cuda.get_device_name(0), "free_bytes": free, "total_bytes": total}


def resolve_training_profile(gpu: dict, profile: str = "auto") -> str:
    """Escolhe head em GPUs de ate 12 GiB; full exige margem para gradientes e Adam."""
    if profile not in {"auto", "head", "full"}:
        raise ValueError("Perfil de treinamento invalido")
    selected = "head" if profile == "auto" and gpu["total_bytes"] <= 12 * 1024**3 else profile
    selected = "full" if selected == "auto" else selected
    if selected == "full" and gpu["free_bytes"] < 12 * 1024**3:
        raise RuntimeError("Perfil full exige 12 GiB livres; use LAYA_TRAIN_PROFILE=head nesta GPU")
    return selected


def token_preflight(directory: Path, base: str = "convaiinnovations/laya") -> dict:
    """Verifica todos os contextos com tokenizer real sem carregar pesos."""
    from huggingface_hub import snapshot_download  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    verify_dataset(directory)
    local = (
        Path(base)
        if Path(base).is_dir()
        else Path(
            snapshot_download(base, allow_patterns=["rl_agent_config.json", "encoder/config.json", "tokenizer/*"])
        )
    )
    config = read_json(local / "rl_agent_config.json")
    encoder = read_json(local / "encoder/config.json")
    tokenizer = AutoTokenizer.from_pretrained(str(local / "tokenizer"))
    agent = SimpleNamespace(
        tok=tokenizer, cfg=config, model=SimpleNamespace(encoder=SimpleNamespace(config=SimpleNamespace(**encoder)))
    )
    budget = 0
    counts = {}
    for name in ("train", "calibration", "selection", "test"):
        rows = read_rows(directory / f"{name}.jsonl")
        counts[name] = len(rows)
        for row in rows:
            _, required, _ = encode_laya_context(agent, json.loads(row["state"]), row["questions"])
            budget = max(budget, required)
    return {
        "max_len": budget,
        "head_max_len": config.get("head_max_len", 192),
        "counts": counts,
        "base_checkpoint": str(local),
        "base_revision": local.name,
    }


def train(
    directory: Path,
    output: Path,
    *,
    dry_run: bool = False,
    base: str = "convaiinnovations/laya",
    profile: str = "auto",
    preset: str = "fast",
    warm_start: Path | None = None,
    completed_epochs: int = 0,
    resume: bool = False,
) -> dict:
    """Resolve o preset e executa somente as epocas restantes com exclusao mutua."""
    import torch  # noqa: PLC0415

    from tools.laya_workflow.run_guard import training_lock  # noqa: PLC0415
    from tools.laya_workflow.training_run import execute_run, schedule, validate_warm_start  # noqa: PLC0415

    if completed_epochs and warm_start is None:
        raise ValueError("Epocas herdadas exigem checkpoint de origem")
    if warm_start and preset != "fast":
        raise ValueError("Warm start legado disponivel somente no preset fast")
    plan = schedule(preset, completed_epochs)
    verify_dataset(directory)
    if (directory / "test_consumed.json").exists():
        raise ValueError("Teste ja consumido: nao ajustar novos candidatos neste dataset")
    if dry_run:
        budget = token_preflight(directory, base)
        if warm_start:
            validate_warm_start(warm_start, Path(budget["base_checkpoint"]), budget)
        return {"dry_run": True, "plan": plan, **budget}
    with training_lock(directory):
        gpu = cuda_preflight(torch)
        selected_profile = resolve_training_profile(gpu, profile)
        print(
            "TRAIN_DEVICE | {} | profile={} | livres={:.2f}GiB".format(
                gpu["device"], selected_profile, gpu["free_bytes"] / 1024**3
            ),
            flush=True,
        )
        print("TRAIN_STAGE | verificacao dos tokens", flush=True)
        budget = token_preflight(directory, base)
        return execute_run(directory, output, budget, gpu, selected_profile, plan, warm_start=warm_start, resume=resume)
