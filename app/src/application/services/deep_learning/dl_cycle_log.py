"""Formatacao compacta de logs do ciclo Deep Learning."""

from src.application.services.deep_learning.dl_gating import resolve_edge
from src.application.services.execution_direction import infer_dl_direction
from src.application.services.log_dedupe import log_info_if_changed
from src.domain.risk.stake_sizing import raw_side_from_metrics


def build_dl_cycle_summary(
    decisions: dict[str, dict],
    *,
    recovery_active: bool,
    pending_loss_total: float,
) -> str:
    """Monta uma unica linha resumo com candidatos e bloqueios do ciclo DL."""
    exec_tokens: list[str] = []
    skip_tokens: list[str] = []
    train_tokens: list[str] = []
    for symbol, entry in decisions.items():
        direction = entry.get("direction")
        metrics = entry.get("metrics") or {}
        if metrics.get("gate_reason") == "training":
            train_tokens.append(symbol)
            continue
        if direction is None:
            gate = metrics.get("gate_reason")
            raw = metrics.get("raw_prob")
            if gate == "data":
                skip_tokens.append(f"{symbol}:data")
            elif gate == "confidence" and raw is not None:
                skip_tokens.append(f"{symbol}:sem_dir:r{float(raw):.2f}")
            else:
                skip_tokens.append(f"{symbol}:sem_dir")
            continue
        conv = float(metrics.get("trade_score", metrics.get("conviction", 0.0)))
        val_acc = float(metrics.get("val_accuracy", 0.0))
        raw = metrics.get("raw_prob")
        gap_s = ""
        if raw is not None:
            gap_s = f":e{resolve_edge(float(raw)):.2f}"
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
    train_part = f" | treino=[{','.join(train_tokens)}]" if train_tokens else ""
    return f"DL | {mode} | exec=[{exec_part}] | skip=[{skip_part}{extra}]{train_part}"


def _best_directional_signal(decisions: dict[str, dict]) -> tuple[str, float] | None:
    """Retorna simbolo e sinal efetivo mais forte entre candidatos com direcao DL."""
    best_symbol = None
    best_score = -1.0
    for symbol, entry in decisions.items():
        metrics = entry.get("metrics") or {}
        if metrics.get("gate_reason") == "training":
            continue
        if infer_dl_direction(entry) is None:
            continue
        score = float(metrics.get("trade_score", metrics.get("conviction", 0.0)))
        raw_side = raw_side_from_metrics(metrics)
        effective = max(score, raw_side)
        if effective > best_score:
            best_score = effective
            best_symbol = symbol
    if best_symbol is None:
        return None
    return best_symbol, best_score


def _abstain_detail(decisions: dict[str, dict]) -> str:
    """Resume simbolos bloqueados com probabilidade bruta quando disponivel."""
    tokens: list[str] = []
    for symbol, entry in decisions.items():
        metrics = entry.get("metrics") or {}
        if metrics.get("gate_reason") == "training":
            continue
        raw = metrics.get("raw_prob")
        gate = metrics.get("gate_reason") or "block"
        if raw is not None:
            tokens.append(f"{symbol}:r{float(raw):.2f}:{gate}")
        else:
            tokens.append(f"{symbol}:{gate}")
    return ",".join(tokens[:4])


def _all_blocked_brief(
    tag: str,
    *,
    blocked: int,
    no_data: int,
    decisions: dict[str, dict],
    train_part: str,
) -> str | None:
    """Monta linha quando todos os simbolos foram bloqueados no ciclo."""
    if blocked != len(decisions) or not decisions:
        return None
    if no_data == len(decisions):
        return f"DL {tag}| sem exec | {no_data} sem dados{train_part}"
    detail = _abstain_detail(decisions)
    suffix = f"aguardando sinal | [{detail}]" if detail else f"{blocked} bloq"
    return f"DL {tag}| sem exec | {suffix}{train_part}"


def build_dl_cycle_brief(
    decisions: dict[str, dict],
    *,
    recovery_active: bool,
) -> str:
    """Linha curta para o console com execucao ativa e contagem de bloqueios."""
    exec_tokens: list[str] = []
    blocked = 0
    no_data = 0
    training = 0
    for symbol, entry in decisions.items():
        direction = entry.get("direction")
        metrics = entry.get("metrics") or {}
        if metrics.get("gate_reason") == "training":
            training += 1
            continue
        if direction is None:
            blocked += 1
            if metrics.get("gate_reason") == "data":
                no_data += 1
            continue
        conv = float(metrics.get("trade_score", metrics.get("conviction", 0.0)))
        if not metrics.get("execute"):
            if infer_dl_direction(entry) is not None:
                exec_tokens.append(f"{symbol}:{direction.name} c={conv:.2f}")
            else:
                blocked += 1
            continue
        exec_tokens.append(f"{symbol}:{direction.name} c={conv:.2f}")
    tag = "REC " if recovery_active else ""
    train_part = f" | {training} treinando" if training else ""
    if exec_tokens:
        head = ",".join(exec_tokens[:2])
        more = f" +{len(exec_tokens) - 2}" if len(exec_tokens) > 2 else ""
        tail = f" | {blocked} bloq" if blocked else ""
        return f"DL {tag}| exec {head}{more}{tail}{train_part}"
    if training == len(decisions):
        return f"DL {tag}| TREINO INICIAL | {training} modelo(s) em treinamento | trades suspensos"
    blocked_msg = _all_blocked_brief(
        tag,
        blocked=blocked,
        no_data=no_data,
        decisions=decisions,
        train_part=train_part,
    )
    if blocked_msg is not None:
        return blocked_msg
    if no_data:
        return f"DL {tag}| sem exec | {no_data} sem dados{train_part}"
    return f"DL {tag}| sem exec | {blocked} bloq{train_part}"


def _brief_key_token(symbol: str, entry: dict) -> tuple[str | None, int, int, int]:
    """Retorna (token, blocked_delta, no_data_delta, training_delta) para a chave de log."""
    metrics = entry.get("metrics") or {}
    if metrics.get("gate_reason") == "training":
        return None, 0, 0, 1
    direction = entry.get("direction")
    if direction is None:
        nd = 1 if metrics.get("gate_reason") == "data" else 0
        return None, 1, nd, 0
    if not metrics.get("execute"):
        if infer_dl_direction(entry) is not None:
            return f"{symbol}:{direction.name}", 0, 0, 0
        return None, 1, 0, 0
    return f"{symbol}:{direction.name}", 0, 0, 0


def build_dl_cycle_brief_key(
    decisions: dict[str, dict],
    *,
    recovery_active: bool,
) -> str:
    """Chave para deduplicacao de logs curtos desconsiderando scores volateis."""
    exec_tokens: list[str] = []
    blocked = 0
    no_data = 0
    training = 0
    for symbol, entry in decisions.items():
        token, b_d, nd_d, t_d = _brief_key_token(symbol, entry)
        if token:
            exec_tokens.append(token)
        blocked += b_d
        no_data += nd_d
        training += t_d
    tag = "REC " if recovery_active else ""
    train_part = f" | {training} treinando" if training else ""
    if exec_tokens:
        head = ",".join(exec_tokens[:2])
        more = f" +{len(exec_tokens) - 2}" if len(exec_tokens) > 2 else ""
        tail = f" | {blocked} bloq" if blocked else ""
        return f"DL {tag}| exec {head}{more}{tail}{train_part}"
    if training == len(decisions):
        return f"DL {tag}| TREINO INICIAL | {training} modelo(s) em treinamento | trades suspensos"
    blocked_msg = _all_blocked_brief(
        tag,
        blocked=blocked,
        no_data=no_data,
        decisions=decisions,
        train_part=train_part,
    )
    if blocked_msg is not None:
        for _, entry in decisions.items():
            metrics = entry.get("metrics") or {}
            raw = metrics.get("raw_prob")
            if raw is not None:
                blocked_msg = blocked_msg.replace(f":r{float(raw):.2f}", "")
        return blocked_msg
    if no_data:
        return f"DL {tag}| sem exec | {no_data} sem dados{train_part}"
    return f"DL {tag}| sem exec | {blocked} bloq{train_part}"


def log_dl_cycle_summary(
    logger,
    decisions: dict[str, dict],
    *,
    recovery_active: bool,
    pending_loss_total: float,
    orch=None,
) -> None:
    """Registra resumo detalhado em DEBUG e linha curta em INFO sem repetir conteudo."""
    logger.debug(
        build_dl_cycle_summary(
            decisions,
            recovery_active=recovery_active,
            pending_loss_total=pending_loss_total,
        )
    )
    brief = build_dl_cycle_brief(
        decisions,
        recovery_active=recovery_active,
    )
    if orch is None:
        logger.info(brief)
        return
    key_brief = build_dl_cycle_brief_key(
        decisions,
        recovery_active=recovery_active,
    )
    log_info_if_changed(orch, logger, "dl_brief", key_brief, "%s", brief)
