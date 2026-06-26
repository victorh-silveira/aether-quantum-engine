"""Formatacao compacta de linhas curtas do ciclo Deep Learning."""

from src.application.services.execution_direction_resolver import infer_dl_direction, is_technically_blocked


def _format_bias_token(symbol: str, entry: dict) -> str | None:
    """Formata token de ajuste direcional quando exec difere do DL."""
    metrics = entry.get("metrics") or {}
    dl_dir = metrics.get("dl_direction") or (entry.get("direction").name if entry.get("direction") else None)
    exec_dir = metrics.get("exec_direction")
    if not dl_dir or not exec_dir or str(dl_dir).upper() == str(exec_dir).upper():
        hint = metrics.get("direction_hint")
        if not hint:
            return None
        conv = float(metrics.get("trade_score", metrics.get("conviction", 0.0)))
        side = dl_dir or exec_dir
        return f"{symbol}:{side} c={conv:.2f}({hint})"
    conv = float(metrics.get("trade_score", metrics.get("conviction", 0.0)))
    hint = metrics.get("direction_hint") or "flip"
    return f"{symbol}:{dl_dir} c={conv:.2f}->{exec_dir}({hint})"


def _abstain_detail(decisions: dict[str, dict]) -> str:
    """Resume simbolos bloqueados tecnicamente quando disponivel."""
    tokens: list[str] = []
    for symbol, entry in decisions.items():
        metrics = entry.get("metrics") or {}
        if metrics.get("gate_reason") == "training":
            continue
        if not is_technically_blocked(entry) and infer_dl_direction(entry) is not None:
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


def _format_brief_token(symbol: str, direction, conv: float, *, suffix: str | None = None) -> str:
    """Formata token de simbolo para linha curta do ciclo DL."""
    token = f"{symbol}:{direction.name} c={conv:.2f}"
    if suffix:
        return f"{token}{suffix}"
    return token


def _brief_cycle_counts(decisions: dict[str, dict]) -> tuple[list[str], list[str], int, int, int]:
    """Separa candidatos tecnicos, ajustes direcionais e contadores de bloqueio."""
    exec_tokens: list[str] = []
    bias_tokens: list[str] = []
    blocked = 0
    no_data = 0
    training = 0
    for symbol, entry in decisions.items():
        metrics = entry.get("metrics") or {}
        if metrics.get("gate_reason") == "training":
            training += 1
            continue
        if is_technically_blocked(entry):
            blocked += 1
            if metrics.get("gate_reason") == "data":
                no_data += 1
            continue
        if infer_dl_direction(entry) is None:
            blocked += 1
            continue
        direction = entry.get("direction") or infer_dl_direction(entry)
        conv = float(metrics.get("trade_score", metrics.get("conviction", 0.0)))
        if metrics.get("execute", True):
            exec_tokens.append(_format_brief_token(symbol, direction, conv))
        bias = _format_bias_token(symbol, entry)
        if bias:
            bias_tokens.append(bias)
    return exec_tokens, bias_tokens, blocked, no_data, training


def _join_brief_tokens(tokens: list[str], label: str, *, limit: int = 2) -> str:
    """Junta tokens rotulados para a linha curta do ciclo."""
    if not tokens:
        return ""
    head = ",".join(tokens[:limit])
    more = f" +{len(tokens) - limit}" if len(tokens) > limit else ""
    return f"{label} {head}{more}"


def build_dl_cycle_brief(
    decisions: dict[str, dict],
    *,
    recovery_active: bool,
) -> str:
    """Linha curta para o console com execucao ativa e ajustes direcionais."""
    exec_tokens, bias_tokens, blocked, no_data, training = _brief_cycle_counts(decisions)
    tag = "REC " if recovery_active else ""
    train_part = f" | {training} treinando" if training else ""
    if exec_tokens or bias_tokens:
        parts = [_join_brief_tokens(exec_tokens, "exec"), _join_brief_tokens(bias_tokens, "bias")]
        body = " | ".join(part for part in parts if part)
        tail = f" | {blocked} bloq" if blocked else ""
        return f"DL {tag}| {body}{tail}{train_part}"
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
    if is_technically_blocked(entry):
        nd = 1 if metrics.get("gate_reason") == "data" else 0
        return None, 1, nd, 0
    if infer_dl_direction(entry) is None:
        return None, 1, 0, 0
    direction = entry.get("direction") or infer_dl_direction(entry)
    if metrics.get("execute", True):
        return f"exec:{symbol}:{direction.name}", 0, 0, 0
    bias = _format_bias_token(symbol, entry)
    if bias:
        return f"bias:{bias}", 0, 0, 0
    gate = str(metrics.get("gate_reason") or "block")
    return f"sinal:{symbol}:{direction.name}:{gate}", 0, 0, 0


def build_dl_cycle_brief_key(
    decisions: dict[str, dict],
    *,
    recovery_active: bool,
) -> str:
    """Chave para deduplicacao de logs curtos desconsiderando scores volateis."""
    exec_tokens: list[str] = []
    bias_tokens: list[str] = []
    signal_tokens: list[str] = []
    blocked = 0
    no_data = 0
    training = 0
    for symbol, entry in decisions.items():
        token, b_d, nd_d, t_d = _brief_key_token(symbol, entry)
        if token:
            if token.startswith("exec:"):
                exec_tokens.append(token[5:])
            elif token.startswith("bias:"):
                bias_tokens.append(token[5:])
            elif token.startswith("sinal:"):
                signal_tokens.append(token[6:])
        blocked += b_d
        no_data += nd_d
        training += t_d
    tag = "REC " if recovery_active else ""
    train_part = f" | {training} treinando" if training else ""
    if exec_tokens or bias_tokens or signal_tokens:
        parts = [
            _join_brief_tokens(exec_tokens, "exec"),
            _join_brief_tokens(bias_tokens, "bias"),
            _join_brief_tokens(signal_tokens, "sinal"),
        ]
        body = " | ".join(part for part in parts if part)
        tail = f" | {blocked} bloq" if blocked else ""
        return f"DL {tag}| {body}{tail}{train_part}"
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
