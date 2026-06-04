"""Formatacao compacta de logs do ciclo Deep Learning."""

from src.application.services.deep_learning.dl_gating import calibration_gap


def build_dl_cycle_summary(
    decisions: dict[str, dict],
    *,
    recovery_active: bool,
    pending_loss_total: float,
) -> str:
    """Monta uma unica linha resumo com candidatos e bloqueios do ciclo DL."""
    exec_tokens: list[str] = []
    skip_tokens: list[str] = []
    for symbol, entry in decisions.items():
        direction = entry.get("direction")
        metrics = entry.get("metrics") or {}
        if direction is None:
            gate = metrics.get("gate_reason")
            raw = metrics.get("raw_prob")
            if gate == "data":
                skip_tokens.append(f"{symbol}:data")
            elif gate == "direction_margin" and raw is not None:
                skip_tokens.append(f"{symbol}:sem_dir:r{float(raw):.2f}")
            else:
                skip_tokens.append(f"{symbol}:sem_dir")
            continue
        conv = float(metrics.get("trade_score", metrics.get("conviction", 0.0)))
        val_acc = float(metrics.get("val_accuracy", 0.0))
        raw = metrics.get("raw_prob")
        gap_s = ""
        if raw is not None:
            gap_s = f":g{calibration_gap(conv, float(raw)):.2f}"
        label = f"{symbol}:{direction.name}:{conv:.2f}:v{val_acc:.2f}{gap_s}"
        if metrics.get("execute"):
            suffix = ""
            if metrics.get("bypass_val_acc"):
                suffix = ":bypass"
            elif recovery_active:
                suffix = ":rec"
            exec_tokens.append(f"{label}{suffix}")
        else:
            gate = metrics.get("gate_reason") or "block"
            edge = float(metrics.get("edge", 0.0))
            raw = float(metrics.get("raw_prob", metrics.get("raw_conviction", conv)))
            brier = float(metrics.get("val_brier", 0.0))
            ece = float(metrics.get("val_ece", 0.0))
            skip_tokens.append(
                f"{symbol}:{gate}:r{raw:.2f}:c{conv:.2f}:e{edge:.2f}:v{val_acc:.2f}:b{brier:.2f}:x{ece:.2f}"
            )
    mode = f"RECOVERY pend=${pending_loss_total:.0f}" if recovery_active else "NORMAL"
    exec_part = ",".join(exec_tokens) if exec_tokens else "none"
    skip_part = ",".join(skip_tokens[:5]) if skip_tokens else "-"
    extra = f" +{len(skip_tokens) - 5}" if len(skip_tokens) > 5 else ""
    return f"DL | {mode} | exec=[{exec_part}] | skip=[{skip_part}{extra}]"


def build_dl_cycle_brief(
    decisions: dict[str, dict],
    *,
    recovery_active: bool,
) -> str:
    """Linha curta para o console com execucao ativa e contagem de bloqueios."""
    exec_tokens: list[str] = []
    blocked = 0
    no_data = 0
    for symbol, entry in decisions.items():
        direction = entry.get("direction")
        metrics = entry.get("metrics") or {}
        if direction is None:
            blocked += 1
            if metrics.get("gate_reason") == "data":
                no_data += 1
            continue
        if not metrics.get("execute"):
            blocked += 1
            continue
        conv = float(metrics.get("trade_score", metrics.get("conviction", 0.0)))
        exec_tokens.append(f"{symbol}:{direction.name} c={conv:.2f}")
    tag = "REC " if recovery_active else ""
    if exec_tokens:
        head = ",".join(exec_tokens[:2])
        more = f" +{len(exec_tokens) - 2}" if len(exec_tokens) > 2 else ""
        tail = f" | {blocked} bloq" if blocked else ""
        return f"DL {tag}| exec {head}{more}{tail}"
    if no_data:
        return f"DL {tag}| sem exec | {no_data} sem dados"
    return f"DL {tag}| sem exec | {blocked} bloq"


def log_dl_cycle_summary(
    logger,
    decisions: dict[str, dict],
    *,
    recovery_active: bool,
    pending_loss_total: float,
) -> None:
    """Registra resumo detalhado em DEBUG e linha curta em INFO."""
    logger.debug(
        build_dl_cycle_summary(
            decisions,
            recovery_active=recovery_active,
            pending_loss_total=pending_loss_total,
        )
    )
    logger.info(
        build_dl_cycle_brief(
            decisions,
            recovery_active=recovery_active,
        )
    )
