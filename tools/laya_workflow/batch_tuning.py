"""Mede lotes reais sem incorporar aprendizado do benchmark ao treinamento."""

import time

import torch

from tools.laya_workflow.training_state import restore_rng, rng_state


def choose_batch(probe) -> tuple[int, list[dict]]:
    """Escolhe o maior lote de 1/2/4 que respeite a margem de memoria."""
    selected = 0
    records = []
    for size in (1, 2, 4):
        try:
            metrics = probe(size)
        except torch.OutOfMemoryError:
            records.append({"batch": size, "oom": True})
            break
        records.append(dict(metrics, batch=size))
        if metrics["free_margin_bytes"] < 1024**3:
            break
        selected = size
    if selected == 0:
        raise RuntimeError("Nenhum lote deixou 1 GiB de margem; libere memoria da GPU")
    return selected, records


def tune_batch(model, encoded: list, config, device) -> tuple[int, list[dict]]:
    """Mede backward e Adam reais restaurando parametros, buffers e RNGs ao final."""
    from tools.laya_workflow.fast_backend import batch_loss  # noqa: PLC0415

    random_state = rng_state()
    probe_items = sorted(encoded, key=lambda item: len(item["ids"]), reverse=True)[:4]
    saved = {name: param.detach().cpu().clone() for name, param in model.named_parameters() if param.requires_grad}
    buffers = {
        name: value.detach().cpu().clone() for name, value in model.named_buffers() if not name.startswith("encoder.")
    }

    def restore():
        """Descarta todos os efeitos do ensaio sobre a cabeca e seus buffers."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in saved:
                    param.copy_(saved[name].to(device))
            for name, value in model.named_buffers():
                if name in buffers:
                    value.copy_(buffers[name].to(device))
        model.zero_grad(set_to_none=True)
        restore_rng(random_state)
        torch.cuda.empty_cache()

    def probe(size):
        """Cronometra tres microbatches e inclui alocacoes do otimizador."""
        restore()
        torch.cuda.synchronize()
        free, _ = torch.cuda.mem_get_info()
        reserved = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        optimizer = torch.optim.AdamW([param for param in model.parameters() if param.requires_grad], lr=config.head_lr)
        scaler = torch.amp.GradScaler("cuda")
        started = time.perf_counter()
        try:
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                loss = batch_loss(model, probe_items[:size], config, device, epoch=0)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            peak = torch.cuda.max_memory_reserved()
            return {
                "seconds": seconds,
                "examples_per_second": 3 * size / seconds,
                "free_margin_bytes": free - max(0, peak - reserved),
            }
        finally:
            del optimizer, scaler
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

    try:
        selected, records = choose_batch(probe)
        print(f"TRAIN_BATCH | micro_batch={selected} grad_accum={64 // selected} effective_batch=64", flush=True)
        return selected, records
    finally:
        restore()
