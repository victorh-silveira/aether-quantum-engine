"""Checkpoint atomico do modelo e do estado exato de otimizacao."""

import random
from pathlib import Path

import numpy as np
import torch

from tools.laya_workflow.artifacts import read_json, write_json


def rng_state() -> dict:
    """Captura estados aleatorios serializaveis com weights_only=True."""
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
    }


def restore_rng(state: dict) -> None:
    """Restaura RNGs apos carregar pesos, otimizador e caches."""
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    array = state["numpy"]
    np.random.set_state((array[0], np.asarray(array[1], dtype=np.uint32), *array[2:]))


def save_training_checkpoint(target: Path, model, tok, cfg: dict, state: dict) -> None:
    """Publica geracao completa antes de trocar o ponteiro da retomada."""
    from laya.train import save_checkpoint  # noqa: PLC0415

    target.mkdir(parents=True, exist_ok=True)
    name = f"epoch-{state['epoch']}-sample-{state['position']}"
    generation = target / "resume" / name
    generation.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, tok, cfg, str(generation))
    # A copia fp32 e necessaria: o checkpoint publico Laya arredonda pesos para fp16.
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    frozen = not any(name.startswith("encoder.") for name in trainable)
    payload = dict(
        state,
        frozen_encoder=frozen,
        model={
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
            if not frozen or not key.startswith("encoder.")
        },
    )
    temporary = generation / "state.pt.tmp"
    torch.save(payload, temporary)
    temporary.replace(generation / "state.pt")
    pointer = target / "resume.json.tmp"
    write_json(pointer, {"generation": name})
    pointer.replace(target / "resume.json")


def load_training_checkpoint(target: Path) -> tuple[Path, dict]:
    """Le somente uma geracao publicada, nunca um checkpoint parcialmente escrito."""
    name = read_json(target / "resume.json")["generation"]
    if Path(name).name != name:
        raise ValueError("Ponteiro de retomada invalido")
    generation = target / "resume" / name
    return generation, torch.load(generation / "state.pt", map_location="cpu", weights_only=True)
