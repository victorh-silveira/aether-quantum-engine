"""Execucao dos presets, heranca de pesos e calibracao temporal reservada."""

import gc
from pathlib import Path

from src.infrastructure.inference.laya_checkpoint import checkpoint_digest
from tools.laya_workflow.artifacts import file_digest, read_json, read_rows, write_json
from tools.laya_workflow.batch_tuning import tune_batch
from tools.laya_workflow.evaluation import score_rows
from tools.laya_workflow.fast_backend import collect_calibration_records, configure_model, run_training
from tools.laya_workflow.training_state import load_training_checkpoint


def schedule(preset: str, completed_epochs: int = 0) -> dict:
    """Resolve o experimento sem alterar dados ou criterios de promocao."""
    if preset not in {"fast", "complete"}:
        raise ValueError("Preset invalido")
    epochs = 2 if preset == "fast" else 4
    if not 0 <= completed_epochs < epochs:
        raise ValueError("Quantidade de epocas herdadas invalida")
    return {
        "preset": preset,
        "epochs": epochs,
        "seeds": [17] if preset == "fast" else [17, 42, 73],
        "inherited_epochs": completed_epochs,
        "remaining_epochs": epochs - completed_epochs,
    }


def validate_warm_start(source: Path, base: Path, budget: dict) -> str:
    """Confere arquitetura e vocabulario; nao inventa historico de otimizacao legado."""
    if not source.is_dir():
        raise ValueError(f"Checkpoint legado incompleto: diretorio ausente ({source})")
    if source.resolve() == base.resolve():
        raise ValueError("Warm start precisa apontar para os pesos treinados")
    for rel in ("model.safetensors", "rl_agent_config.json", "encoder/config.json", "tokenizer/tokenizer.json"):
        required = source / rel
        if not required.is_file() or required.stat().st_size == 0:
            raise ValueError(f"Checkpoint legado incompleto: {rel}")
    cfg, reference = read_json(source / "rl_agent_config.json"), read_json(base / "rl_agent_config.json")
    encoder, base_encoder = read_json(source / "encoder/config.json"), read_json(base / "encoder/config.json")
    for key in ("encoder", "head_layers", "head_max_len", "max_prefixes"):
        if cfg.get(key) != reference.get(key):
            raise ValueError(f"Checkpoint legado incompativel: {key}")
    for key in ("model_type", "hidden_size", "num_hidden_layers", "vocab_size"):
        if encoder.get(key) != base_encoder.get(key):
            raise ValueError(f"Encoder legado incompativel: {key}")
    if cfg.get("max_len", 0) < budget["max_len"]:
        raise ValueError("Contexto legado menor que o dataset atual")
    tok_src = {k: v for k, v in read_json(source / "tokenizer/tokenizer.json").items() if k != "truncation"}
    tok_ref = {k: v for k, v in read_json(base / "tokenizer/tokenizer.json").items() if k != "truncation"}
    if tok_src != tok_ref:
        raise ValueError("Tokenizer legado incompativel")
    return checkpoint_digest(source)


def write_run(path: Path, value: dict) -> None:
    """Atualiza o manifesto da execucao sem publicar JSON parcial."""
    temporary = path.with_suffix(".tmp")
    write_json(temporary, value)
    temporary.replace(path)


def execute_run(
    directory: Path,
    output: Path,
    budget: dict,
    gpu: dict,
    profile: str,
    plan: dict,
    *,
    warm_start: Path | None = None,
    resume: bool = False,
) -> dict:
    """Treina somente as epocas restantes e seleciona fora do teste final."""
    import torch  # noqa: PLC0415
    from huggingface_hub import snapshot_download  # noqa: PLC0415
    from laya import load  # noqa: PLC0415
    from laya.calibrate import fit_temperature_map  # noqa: PLC0415
    from laya.train import (  # noqa: PLC0415
        TrainConfig,
        encode_item,
        items_from_rows,
        load_checkpoint,
        save_checkpoint,
    )

    local = Path(budget["base_checkpoint"])
    if not (local / "model.safetensors").exists():
        local = Path(
            snapshot_download(
                "convaiinnovations/laya",
                revision=budget["base_revision"],
                allow_patterns=["model.safetensors", "rl_agent_config.json", "encoder/*", "tokenizer/*"],
            )
        )
    source_digest = validate_warm_start(warm_start, local, budget) if warm_start else None
    identity = {
        "dataset_sha256": file_digest(directory / "manifest.json"),
        "plan": plan,
        "profile": profile,
        "warm_start": str(warm_start.resolve()) if warm_start else None,
        "source_checkpoint_sha256": source_digest,
        "optimizer_restarted": bool(warm_start),
        "base_revision": budget["base_revision"],
    }
    if output.exists():
        if not resume:
            raise ValueError("Diretorio ja existe; use LAYA_ARGS=--resume para retomada ou outro LAYA_RUN")
        previous = read_json(output / "run.json")
        if any(previous.get(key) != value for key, value in identity.items()):
            raise ValueError("Dataset, pesos de origem ou configuracao mudaram; retomada recusada")
        manifest = previous
    else:
        if resume:
            raise ValueError("Execucao inexistente para retomar")
        output.mkdir(parents=True)
        manifest = dict(identity, batches={})
        write_run(output / "run.json", manifest)
    print(
        f"TRAIN_PLAN | preset={plan['preset']} seeds={plan['seeds']} epochs={plan['epochs']} "
        f"inherited={plan['inherited_epochs']} remaining={plan['remaining_epochs']} optimizer_restarted={bool(warm_start)}",
        flush=True,
    )
    cached = None
    candidates = []
    for seed in plan["seeds"]:
        target = output / f"seed-{seed}"
        state = None
        checkpoint = warm_start or local
        if resume and (target / "resume.json").exists():
            _, state = load_training_checkpoint(target)
        model, tok, cfg = load_checkpoint(str(checkpoint))
        cfg = dict(cfg, max_len=budget["max_len"], head_max_len=budget["head_max_len"])
        if cached is None:
            print("TRAIN_STAGE | preparacao dos exemplos tokenizados", flush=True)
            items = {}
            for name in ("train", "calibration"):
                items[name], skipped = items_from_rows(
                    tok, read_rows(directory / f"{name}.jsonl"), cfg["max_len"], cfg["head_max_len"]
                )
                if skipped or not items[name]:
                    raise ValueError(f"Exemplos de {name} rejeitados: {skipped}")
            encoded = [encode_item(tok, item, cfg["max_len"], cfg["head_max_len"]) for item in items["train"]]
            cached = (encoded, items["calibration"])
            del items
        encoded, calibration_items = cached
        options = TrainConfig(
            epochs=plan["epochs"],
            micro_batch=1,
            grad_accum=64,
            encoder_lr=0.000025,
            head_lr=0.0001,
            loss="rlcd",
            amp=True,
            gradient_checkpointing=True,
            freeze_encoder=profile == "head",
            seed=seed,
            max_len=cfg["max_len"],
            head_max_len=cfg["head_max_len"],
        )
        options.pad_token_id = tok.pad_token_id
        configure_model(model, options, torch.device("cuda"))
        if str(seed) not in manifest["batches"]:
            if profile == "head":
                print("TRAIN_STAGE | medicao de lotes 1/2/4", flush=True)
                micro, measurements = tune_batch(model, encoded, options, torch.device("cuda"))
            else:
                micro, measurements = 1, []
            manifest["batches"][str(seed)] = {"micro_batch": micro, "measurements": measurements}
            write_run(output / "run.json", manifest)
        options.micro_batch = manifest["batches"][str(seed)]["micro_batch"]
        options.grad_accum = 64 // options.micro_batch
        if state and (
            state["micro_batch"] != options.micro_batch or state["epochs"] != options.epochs or state["seed"] != seed
        ):
            raise ValueError("Parametros de retomada incompativeis")
        print(f"TRAIN_STAGE | treinamento seed={seed} micro_batch={options.micro_batch}", flush=True)
        history = run_training(
            model,
            tok,
            encoded,
            options,
            torch.device("cuda"),
            target=target,
            cfg=cfg,
            start_epoch=plan["inherited_epochs"],
            state=state,
        )
        resume_mode = "exact_resume" if state is not None else ("legacy_warm_start" if warm_start else "fresh")
        torch.cuda.empty_cache()
        print(f"TRAIN_STAGE | calibracao seed={seed} samples={len(calibration_items)}", flush=True)
        records = collect_calibration_records(
            model,
            tok,
            calibration_items,
            torch.device("cuda"),
            cfg["max_len"],
            cfg["head_max_len"],
            batch_size=options.micro_batch,
            amp=bool(options.amp),
        )
        fitted = fit_temperature_map(records)
        cfg.update(temperature=fitted["temperature"], temperature_by_options=fitted["temperature_by_options"])
        print(f"TRAIN_STAGE | gravacao seed={seed}", flush=True)
        save_checkpoint(model, tok, cfg, str(target))
        write_json(target / "questions.json", read_json(directory / "questions.json"))
        write_json(
            target / "training.json",
            dict(
                identity,
                seed=seed,
                resume_mode=resume_mode,
                inherited_epochs=plan["inherited_epochs"],
                executed_epochs=len(history),
                history=history,
                gpu=gpu,
                micro_batch=options.micro_batch,
                grad_accum=options.grad_accum,
                batches=manifest["batches"][str(seed)],
                **budget,
            ),
        )
        del model, state, records
        gc.collect()
        torch.cuda.empty_cache()
        print(f"TRAIN_STAGE | selecao seed={seed}", flush=True)
        agent = load(str(target), device="cuda", vegaml=False)
        metrics = score_rows(agent, read_rows(directory / "selection.jsonl"), batch_size=options.micro_batch)
        candidates.append(
            {
                "candidate": target.name,
                "metrics": metrics,
                "micro_batch": options.micro_batch,
                "resume_mode": resume_mode,
                "checkpoint_sha256": checkpoint_digest(target),
            }
        )
        del agent
        gc.collect()
        torch.cuda.empty_cache()
    selected = max(candidates, key=lambda item: (item["metrics"]["accuracy"], -item["metrics"]["brier"]))
    result = {
        **selected,
        "candidates": candidates,
        "batches": manifest["batches"],
        "resume_modes": {item["candidate"]: item["resume_mode"] for item in candidates},
        "gpu": gpu,
        **identity,
        **budget,
    }
    print("TRAIN_STAGE | gravacao selection.json", flush=True)
    write_json(output / "selection.json", result)
    print("TRAIN_DONE | selection.json gravado; proximo comando: make app-evaluate", flush=True)
    return result
