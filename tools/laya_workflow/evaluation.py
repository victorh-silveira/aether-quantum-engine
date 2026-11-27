"""Metricas fora da amostra, selecao e promocao de checkpoints Laya."""

import json
import math
import time
from pathlib import Path

from src.infrastructure.inference.laya_checkpoint import checkpoint_digest
from src.infrastructure.inference.laya_context import encode_laya_context, validate_laya_usage
from src.infrastructure.inference.laya_decision_adapter import _extract_choice_probs
from tools.laya_workflow.artifacts import file_digest, read_json, read_rows, verify_dataset, write_json


def predict_rows(agent, rows: list[dict], batch_size: int):
    """Agrupa forwards preservando ordem, contexto e validacao de cada resposta."""
    if batch_size < 1:
        raise ValueError("Lote de avaliacao invalido")
    started = last = time.perf_counter()
    for offset in range(0, len(rows), batch_size):
        chunk = rows[offset : offset + batch_size]
        states = [json.loads(row["state"]) for row in chunk]
        questions = chunk[0]["questions"]
        if any(row["questions"] != questions for row in chunk):
            raise ValueError("Perguntas diferentes no mesmo lote")
        encoded = [encode_laya_context(agent, state, questions) for state in states]
        budget = max(item[1] for item in encoded)
        if batch_size == 1:
            answers = [agent.system_one(encoded[0][0], questions, max_len=budget)]
        else:
            answers = agent.predict_batch(
                [item[0] for item in encoded], questions, batch_size=batch_size, max_len=budget
            )
        if len(answers) != len(chunk):
            raise ValueError("Quantidade de respostas difere do lote")
        yield from zip(chunk, states, answers, strict=True)
        now = time.perf_counter()
        processed = offset + len(chunk)
        if now - last >= 5 or processed == len(rows):
            elapsed = now - started
            rate = processed / elapsed if elapsed > 0 else 0.0
            eta = (
                f"{(len(rows) - processed) / rate / 60:.2f}min"
                if processed >= max(batch_size, 1) and rate > 0
                else "calculando"
            )
            print(
                f"EVAL_PROGRESS | samples={processed}/{len(rows)} speed={rate:.2f}/s "
                f"elapsed={elapsed / 60:.2f}min eta={eta}",
                flush=True,
            )
            last = now


def score_rows(agent, rows: list[dict], *, batch_size: int = 1) -> dict:
    """Avalia probabilidade binaria sem misturar magnitude da stake com acerto."""
    correct = calls = 0
    brier = 0.0
    counts = {"CALL": 0, "PUT": 0}
    hits = {"CALL": 0, "PUT": 0}
    for row, state, raw in predict_rows(agent, rows, batch_size):
        validate_laya_usage(raw)
        probabilities = raw.get("answers", {}).get("direction", {}).get("probabilities", {})
        values = [probabilities.get(label) for label in ("CALL", "PUT")]
        if (
            not all(isinstance(p, (int, float)) and math.isfinite(p) and 0 <= p <= 1 for p in values)
            or sum(values) <= 0
        ):
            raise ValueError("Probabilidades invalidas na avaliacao; nao usar fallback como evidencia")
        direction, pc, _, _ = _extract_choice_probs(raw, state.get("indicators", {}))
        gold = row["expected"]["direction"]
        counts[gold] += 1
        hit = direction.name == gold
        hits[gold] += int(hit)
        correct += int(hit)
        calls += int(direction.name == "CALL")
        brier += (pc - int(gold == "CALL")) ** 2
    n = len(rows)
    if not n or not all(counts.values()):
        raise ValueError("Avaliacao exige amostras de ambas as classes")
    recalls = {label: hits[label] / counts[label] for label in counts}
    return {
        "n": n,
        "accuracy": correct / n,
        "majority": max(counts.values()) / n,
        "balanced_accuracy": sum(recalls.values()) / 2,
        "recall": recalls,
        "brier": brier / n,
        "predicted_call_fraction": calls / n,
    }


def promotion_allowed(candidate: dict, baseline: dict) -> bool:
    """Exige todos os criterios confirmados sem selecionar no teste final."""
    return bool(
        candidate["n"] == baseline["n"]
        and candidate["accuracy"] + 1e-12 >= baseline["accuracy"] + 0.01
        and candidate["accuracy"] > candidate["majority"]
        and candidate["balanced_accuracy"] > 0.5
        and min(candidate["recall"].values()) >= 0.45
        and candidate["brier"] <= baseline["brier"]
    )


def evaluate(directory: Path, run: Path, device: str = "cuda", loader=None) -> dict:
    """Consome o teste final uma unica vez para um candidato ja selecionado."""
    import gc  # noqa: PLC0415

    import laya  # noqa: PLC0415

    manifest = verify_dataset(directory)
    selection = read_json(run / "selection.json")
    candidate_path = run / selection["candidate"]
    if selection["dataset_sha256"] != file_digest(directory / "manifest.json"):
        raise ValueError("Selecao pertence a outro dataset")
    if checkpoint_digest(candidate_path) != selection["checkpoint_sha256"]:
        raise ValueError("Candidato alterado apos selecao")
    marker = directory / "test_consumed.json"
    with marker.open("x", encoding="utf-8") as stream:
        json.dump({"candidate": selection["checkpoint_sha256"], "run": str(run.resolve())}, stream)
    loader = loader or (lambda path: laya.load(path, device=device, vegaml=False))
    rows = read_rows(directory / "test.jsonl")
    baseline_agent = loader(selection["base_checkpoint"])
    baseline = score_rows(baseline_agent, rows, batch_size=int(selection.get("micro_batch", 1)))
    del baseline_agent
    gc.collect()
    candidate_agent = loader(str(candidate_path))
    candidate = score_rows(candidate_agent, rows, batch_size=int(selection.get("micro_batch", 1)))
    report = {
        "baseline": baseline,
        "candidate": candidate,
        "approved": promotion_allowed(candidate, baseline),
        "checkpoint_sha256": selection["checkpoint_sha256"],
        "dataset_sha256": selection["dataset_sha256"],
        "label": manifest["label"],
        "realized_profit_claim": False,
    }
    write_json(run / "evaluation.json", report)
    return report


def promote(directory: Path, run: Path, *, allow_unapproved_override: bool = False) -> dict:
    """Emite aprovacao verificavel sem editar settings ou iniciar o motor."""
    verify_dataset(directory)
    selection = read_json(run / "selection.json")
    report = read_json(run / "evaluation.json")
    candidate = run / selection["candidate"]
    digest = checkpoint_digest(candidate)
    is_approved = promotion_allowed(report["candidate"], report["baseline"])
    if (
        (not is_approved and not allow_unapproved_override)
        or digest != report["checkpoint_sha256"]
        or digest != selection["checkpoint_sha256"]
        or report["dataset_sha256"] != file_digest(directory / "manifest.json")
    ):
        raise ValueError("Candidato nao aprovado ou artefatos alterados")
    approval = {
        "approved": bool(is_approved),
        "operator_override": bool(not is_approved and allow_unapproved_override),
        "checkpoint_sha256": digest,
        "evaluation_sha256": file_digest(run / "evaluation.json"),
        "dataset_sha256": report["dataset_sha256"],
    }
    write_json(candidate / "approval.json", approval)
    return {"approved_checkpoint": str(candidate.resolve()), **approval}
