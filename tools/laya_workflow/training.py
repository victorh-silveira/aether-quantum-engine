"""Preparacao de tokens e treinamento CUDA com calibracao temporal externa."""

import gc
import json
from pathlib import Path
from types import SimpleNamespace

from src.infrastructure.inference.laya_checkpoint import checkpoint_digest
from src.infrastructure.inference.laya_context import encode_laya_context
from tools.laya_workflow.artifacts import file_digest, read_json, read_rows, verify_dataset, write_json
from tools.laya_workflow.evaluation import score_rows


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
    directory: Path, output: Path, *, dry_run: bool = False, base: str = "convaiinnovations/laya", profile: str = "auto"
) -> dict:
    """Treina tres seeds sem consumir teste; calibra exclusivamente na faixa reservada."""
    import torch  # noqa: PLC0415

    verify_dataset(directory)
    gpu = None if dry_run else cuda_preflight(torch)
    selected_profile = None if dry_run else resolve_training_profile(gpu, profile)
    if gpu is not None:
        print(
            f"TRAIN_DEVICE | {gpu['device']} | profile={selected_profile} | livres={gpu['free_bytes'] / 1024**3:.2f}GiB",
            flush=True,
        )
    if (directory / "test_consumed.json").exists():
        raise ValueError("Teste ja consumido: nao ajustar novos candidatos neste dataset")
    budget = token_preflight(directory, base)
    if dry_run:
        return {"dry_run": True, **budget}
    if output.exists():
        raise ValueError("Diretorio de treinamento ja existe; preserve os artefatos anteriores")
    output.mkdir(parents=True)
    from huggingface_hub import snapshot_download  # noqa: PLC0415
    from laya import load  # noqa: PLC0415
    from laya.calibrate import fit_temperature_map  # noqa: PLC0415
    from laya.train import (  # noqa: PLC0415
        TrainConfig,
        calibration_records,
        items_from_rows,
        load_checkpoint,
        save_checkpoint,
        train_model,
    )

    local = Path(budget["base_checkpoint"])
    if not (local / "model.safetensors").exists():
        local = Path(
            snapshot_download(
                base,
                revision=budget["base_revision"],
                allow_patterns=["model.safetensors", "rl_agent_config.json", "encoder/*", "tokenizer/*"],
            )
        )
    candidates = []
    for seed in (17, 42, 73):
        model, tok, cfg = load_checkpoint(str(local))
        cfg = dict(cfg, max_len=budget["max_len"], head_max_len=budget["head_max_len"])
        items = {}
        for name in ("train", "calibration"):
            items[name], skipped = items_from_rows(
                tok, read_rows(directory / f"{name}.jsonl"), cfg["max_len"], cfg["head_max_len"]
            )
            if skipped or not items[name]:
                raise ValueError(f"Exemplos de {name} rejeitados: {skipped}")
        target = output / f"seed-{seed}"
        options = TrainConfig(
            epochs=4,
            micro_batch=1,
            grad_accum=64,
            encoder_lr=0.000025,
            head_lr=0.0001,
            loss="rlcd",
            amp=True,
            gradient_checkpointing=True,
            freeze_encoder=selected_profile == "head",
            seed=seed,
            max_len=cfg["max_len"],
            head_max_len=cfg["head_max_len"],
        )
        model.to("cuda")
        history = train_model(
            model,
            tok,
            items["train"],
            options,
            torch.device("cuda"),
            cfg["max_len"],
            cfg["head_max_len"],
            on_epoch_end=lambda _epoch, _loss, m=model, t=tok, c=cfg, p=target: save_checkpoint(
                m, t, c, str(p / "checkpoint_latest")
            ),
        )
        records = calibration_records(
            model, tok, items["calibration"], torch.device("cuda"), cfg["max_len"], cfg["head_max_len"], batch_size=1
        )
        fitted = fit_temperature_map(records)
        cfg.update(temperature=fitted["temperature"], temperature_by_options=fitted["temperature_by_options"])
        save_checkpoint(model, tok, cfg, str(target))
        write_json(target / "questions.json", read_json(directory / "questions.json"))
        write_json(
            target / "training.json",
            {
                "seed": seed,
                "epochs": 4,
                "history": history,
                "gpu": gpu,
                "profile": selected_profile,
                "dataset_sha256": file_digest(directory / "manifest.json"),
                **budget,
            },
        )
        del model, items, records
        gc.collect()
        torch.cuda.empty_cache()
        agent = load(str(target), device="cuda", vegaml=False)
        metrics = score_rows(agent, read_rows(directory / "selection.jsonl"))
        candidates.append(
            {"candidate": target.name, "metrics": metrics, "checkpoint_sha256": checkpoint_digest(target)}
        )
        del agent
        gc.collect()
        torch.cuda.empty_cache()
    selected = max(candidates, key=lambda item: (item["metrics"]["accuracy"], -item["metrics"]["brier"]))
    result = {
        **selected,
        "candidates": candidates,
        "dataset_sha256": file_digest(directory / "manifest.json"),
        "base_checkpoint": str(local),
        "gpu": gpu,
        "profile": selected_profile,
        **budget,
    }
    write_json(output / "selection.json", result)
    return result
