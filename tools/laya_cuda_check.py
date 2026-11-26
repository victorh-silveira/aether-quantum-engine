"""Diagnostico CUDA e provas reais de inferencia/backward Laya, sem enviar ordens."""

import argparse
import asyncio
import json
import math
from datetime import UTC, datetime
from pathlib import Path

from src.domain.indicators.technical_indicators import compute_indicators
from src.domain.models.market_data import Candle
from src.infrastructure.inference.laya_context import encode_laya_context
from src.infrastructure.inference.laya_decision_adapter import _LAYA_QUESTIONS, LayaDecisionAdapter, build_laya_state
from tools.laya_workflow.training import cuda_preflight, resolve_training_profile


def runtime_check(torch) -> dict:
    """Executa multiplicacao real na GPU e confere resultado e sincronizacao."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA indisponivel; execute make app-cuda-install e confira o driver NVIDIA")
    left = torch.ones((32, 32), device="cuda")
    result = left @ left
    torch.cuda.synchronize()
    if not bool(torch.all(result == 32).item()):
        raise RuntimeError("Calculo CUDA retornou resultado invalido")
    free, total = torch.cuda.mem_get_info()
    return {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "tensor_device": str(result.device),
        "free_gib": round(free / 1024**3, 2),
        "total_gib": round(total / 1024**3, 2),
    }


def sample_bars() -> list[Candle]:
    """Gera OHLC deterministico exclusivo do teste, nunca enviado a corretora."""
    bars = []
    for index in range(120):
        epoch = 1764000000 + index * 300
        opening = 4800 + index + math.sin(index) * 2
        close = opening + 0.5
        bars.append(
            Candle("R_10", opening, close + 1, opening - 1, close, datetime.fromtimestamp(epoch, tz=UTC), epoch)
        )
    return bars


async def check_laya(settings: dict, mode: str, torch) -> dict:
    """Comprova forwards na GPU e, opcionalmente, atualiza cabeca somente em memoria."""
    from laya.train import TrainConfig, items_from_rows, train_model  # noqa: PLC0415

    config = settings.get("laya", {})
    adapter = LayaDecisionAdapter(
        model_id=config.get("model_id", "convaiinnovations/laya"),
        device=config.get("device", "auto"),
        approved_checkpoint=config.get("approved_checkpoint", ""),
    )
    agent = adapter._ensure_agent()
    forwards = []

    def assert_cuda(_module, args, kwargs):
        """Impede fallback CPU silencioso durante a prova de operacao."""
        tensors = [value for value in (*args, *kwargs.values()) if isinstance(value, torch.Tensor)]
        if not tensors or any(value.device.type != "cuda" for value in tensors):
            raise RuntimeError("Forward Laya fora da GPU")
        forwards.append("cuda")

    hook = agent.model.register_forward_pre_hook(assert_cuda, with_kwargs=True)
    try:
        bars = sample_bars()
        indicators = compute_indicators(bars).to_dict()
        decision = await adapter.decide("R_10", bars, indicators, settings)
        report = {
            "model_version": decision.model_version,
            "forward_device": forwards[-1],
            "direction": decision.direction.name,
            "probability": decision.probability,
            "latency_ms": round(decision.latency_ms, 2),
        }
        if mode == "training":
            gpu = cuda_preflight(torch)
            profile = resolve_training_profile(gpu)
            state = build_laya_state("R_10", bars, indicators, settings)
            text, budget, _ = encode_laya_context(agent, state, _LAYA_QUESTIONS)
            rows = [
                {"state": text, "questions": _LAYA_QUESTIONS, "expected": {"direction": label}}
                for label in ("CALL", "PUT")
            ]
            items, skipped = items_from_rows(agent.tok, rows, budget, agent.cfg.get("head_max_len", 192))
            if skipped or len(items) != 2:
                raise RuntimeError("Exemplos de diagnostico rejeitados")
            parameters = {
                name: param for name, param in agent.model.named_parameters() if not name.startswith("encoder.")
            }
            before = {name: param.detach().clone() for name, param in parameters.items()}
            history = train_model(
                agent.model,
                agent.tok,
                items,
                TrainConfig(
                    epochs=1,
                    micro_batch=1,
                    grad_accum=1,
                    amp=True,
                    freeze_encoder=profile == "head",
                    gradient_checkpointing=True,
                ),
                torch.device("cuda"),
                budget,
                agent.cfg.get("head_max_len", 192),
            )
            changed = any(not torch.equal(before[name], param.detach()) for name, param in parameters.items())
            if not changed or not all(math.isfinite(value) for value in history):
                raise RuntimeError("Backward/otimizador Laya nao atualizaram a cabeca corretamente")
            report.update(training_profile=profile, training_loss=history, optimizer_updated=True)
        report["cuda_forwards"] = len(forwards)
        return report
    finally:
        hook.remove()


def main() -> None:
    """Executa diagnostico solicitado sem importar o executor financeiro."""
    import torch  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("runtime", "laya", "training"), default="laya")
    parser.add_argument("--settings", type=Path, default=Path("config/settings.json"))
    args = parser.parse_args()
    report = runtime_check(torch)
    if args.mode != "runtime":
        report.update(asyncio.run(check_laya(json.loads(args.settings.read_text(encoding="utf-8")), args.mode, torch)))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
