"""Loop RLCD com lote efetivo de 64, progresso e retomada entre atualizacoes."""

import math
import random
import signal
import time
from contextlib import contextmanager

import torch

from tools.laya_workflow.training_state import restore_rng, rng_state, save_training_checkpoint


def configure_model(model, config, device) -> None:
    """Congela o encoder somente em head e preserva o comportamento de dropout."""
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(not config.freeze_encoder)
    if not config.freeze_encoder and config.gradient_checkpointing:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = config.gradient_checkpointing
    model.to(device).train()
    if config.freeze_encoder:
        model.encoder.eval()


def batch_loss(model, encoded, config, device, *, epoch: int):
    """Reutiliza forward e objetivo RLCD da versao fixada de Laya."""
    from laya.train import _forward, collate_items, rlcd_loss, sigma_at, soft_ce_loss  # noqa: PLC0415

    batch = collate_items([encoded], getattr(config, "pad_token_id", 0))
    logits = _forward(model, batch, device, bool(config.amp), config.freeze_encoder)
    mask, target = batch["marker_mask"].to(device), batch["target"].to(device)
    if config.loss == "rlcd":
        return rlcd_loss(
            logits,
            target,
            mask,
            batch["qtype"].to(device),
            sigma_at(epoch, config.epochs, config.sigma_start, config.sigma_end),
            config.rl_samples,
            config.w_sph,
            config.w_rps,
        )
    return soft_ce_loss(logits, target, mask)


class TrainingProgress:
    """Calcula velocidade e ETA por fase usando amostras observadas."""

    def __init__(self, total: int, *, clock=time.perf_counter, emit=print):
        self.total = total
        self.clock, self.emit = clock, emit
        self.started = self.last_log = clock()
        self.processed = 0

    def update(
        self, samples: int, epoch: int, epochs: int, position: int, epoch_size: int, loss: float | None = None
    ) -> None:
        """Emite no maximo a cada cinco segundos e sempre no fim da epoca."""
        self.processed += samples
        now = self.clock()
        if now - self.last_log < 5 and position != epoch_size:
            return
        elapsed = now - self.started
        rate = self.processed / elapsed if elapsed > 0 else 0
        eta = (
            f"{(self.total - self.processed) / rate / 60:.2f}min" if self.processed >= 64 and rate > 0 else "calculando"
        )
        suffix = f" loss={loss:.2f}" if loss is not None else ""
        self.emit(
            f"TRAIN_PROGRESS | epoch={epoch}/{epochs} samples={position}/{epoch_size} "
            f"speed={rate:.2f}/s elapsed={elapsed / 60:.2f}min eta={eta}{suffix}",
            flush=True,
        )
        self.last_log = now


@contextmanager
def controlled_stop():
    """Adia Ctrl+C ate concluir uma atualizacao atomica do otimizador."""
    stopped = [False]
    previous = signal.getsignal(signal.SIGINT)

    def request_stop(_signum, _frame):
        """Solicita checkpoint sem interromper Adam no meio de seus parametros."""
        stopped[0] = True

    signal.signal(signal.SIGINT, request_stop)
    try:
        yield stopped
    finally:
        signal.signal(signal.SIGINT, previous)


def run_training(
    model, tok, encoded, config, device, *, target, cfg: dict, start_epoch: int = 0, state: dict | None = None
):
    """Treina exemplos pre-encodificados e retoma exatamente em fronteiras de 64 amostras."""
    configure_model(model, config, device)
    groups = [
        {
            "params": [param for name, param in model.named_parameters() if not name.startswith("encoder.")],
            "lr": config.head_lr,
        }
    ]
    if not config.freeze_encoder:
        groups.insert(0, {"params": model.encoder.parameters(), "lr": config.encoder_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=config.weight_decay)
    first_epoch = state["origin_epoch"] if state else start_epoch
    updates = math.ceil(len(encoded) / 64) * (config.epochs - first_epoch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, updates), eta_min=config.min_lr)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config.amp) and device.type == "cuda")
    history = list(state["history"]) if state else []
    current_epoch = state["epoch"] if state else start_epoch
    position = state["position"] if state else 0
    if state:
        missing, unexpected = model.load_state_dict(state["model"], strict=False)
        if unexpected or any(not state.get("frozen_encoder") or not key.startswith("encoder.") for key in missing):
            raise ValueError("Pesos de retomada incompativeis")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        restore_rng(state["rng"])
    else:
        torch.manual_seed(config.seed)
        random.seed(config.seed)
    remaining = (config.epochs - current_epoch) * len(encoded) - position
    progress = TrainingProgress(remaining)
    parameters = [param for param in model.parameters() if param.requires_grad]

    def checkpoint(epoch, sample_position, epoch_loss):
        """Persiste apenas quando nao ha gradientes parcialmente acumulados."""
        save_training_checkpoint(
            target,
            model,
            tok,
            cfg,
            {
                "origin_epoch": first_epoch,
                "epoch": epoch,
                "position": sample_position,
                "history": history,
                "epoch_loss": epoch_loss,
                "rng": rng_state(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "micro_batch": config.micro_batch,
                "epochs": config.epochs,
                "seed": config.seed,
            },
        )

    with controlled_stop() as stopped:
        for epoch in range(current_epoch, config.epochs):
            order = list(range(len(encoded)))
            random.Random(config.seed + epoch).shuffle(order)  # noqa: S311
            loss_sum = torch.tensor(
                float(state.get("epoch_loss", 0)) if state and epoch == current_epoch else 0.0, device=device
            )
            begin = position if epoch == current_epoch else 0
            for offset in range(begin, len(order), 64):
                window = order[offset : offset + 64]
                optimizer.zero_grad(set_to_none=True)
                window_loss = torch.zeros((), device=device)
                for index in range(0, len(window), config.micro_batch):
                    chunk = [encoded[item] for item in window[index : index + config.micro_batch]]
                    loss = batch_loss(model, chunk, config, device, epoch=epoch)
                    weighted = loss * (len(chunk) / len(window))
                    scaler.scale(weighted).backward()
                    window_loss += loss.detach() * len(chunk)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, config.grad_clip)
                scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= scale:
                    scheduler.step()
                loss_sum += window_loss
                completed = offset + len(window)
                progress.update(len(window), epoch + 1, config.epochs, completed, len(order))
                if stopped[0]:
                    optimizer.zero_grad(set_to_none=True)
                    checkpoint(epoch, completed, float(loss_sum.item()))
                    print("TRAIN_PAUSED | checkpoint salvo; retome com make app-train LAYA_ARGS=--resume", flush=True)
                    raise KeyboardInterrupt
            mean = float(loss_sum.item()) / len(order)
            if not math.isfinite(mean):
                raise RuntimeError("Treinamento produziu loss nao finita")
            history.append(mean)
            checkpoint(epoch + 1, 0, 0.0)
            print(f"TRAIN_EPOCH | epoch={epoch + 1}/{config.epochs} mean_loss={mean:.2f}", flush=True)
    model.eval()
    return history


@torch.no_grad()
def collect_calibration_records(
    model,
    tok,
    items,
    device,
    max_len: int,
    head_max_len: int,
    *,
    batch_size: int = 1,
    amp: bool = True,
    clock=time.perf_counter,
    emit=print,
) -> list:
    """Coleta logits de calibracao com AMP opcional e progresso sem paging em FP32."""
    from laya.train import _forward, collate_items, encode_item  # noqa: PLC0415

    model.eval()
    records = []
    started = last = clock()
    use_amp = bool(amp) and device.type == "cuda"
    pad_id = getattr(tok, "pad_token_id", 0)
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        batch = collate_items([[encode_item(tok, it, max_len, head_max_len) for it in chunk]], pad_id)
        logits = _forward(model, batch, device, amp=use_amp, detach_encoder=False).cpu().numpy()
        for row, it in zip(logits, chunk, strict=True):
            records.append((it["qtype"], row[: it["k"]], it["target"], it["k"]))
        now = clock()
        processed = start + len(chunk)
        if now - last >= 5 or processed == len(items):
            elapsed = now - started
            rate = processed / elapsed if elapsed > 0 else 0.0
            eta = (
                f"{(len(items) - processed) / rate / 60:.2f}min"
                if processed >= max(batch_size, 1) and rate > 0
                else "calculando"
            )
            emit(
                f"CALIB_PROGRESS | samples={processed}/{len(items)} speed={rate:.2f}/s "
                f"elapsed={elapsed / 60:.2f}min eta={eta}",
                flush=True,
            )
            last = now
    return records
