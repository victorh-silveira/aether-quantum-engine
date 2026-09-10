"""Peso de amostras de treino a partir de resultados reais de trades."""

import time

from src.domain.config_knobs import merge_settings_block, require_int, require_keys


_SESSION_PAUSE_KEYS = (
    "session_max_losses_in_window",
    "session_window_trades",
    "session_pause_cycles",
)


def _symbol_history(orch, symbol: str) -> list[bool]:
    """Retorna historico de wins/losses registrados para o simbolo."""
    flags = getattr(orch, "_dl_outcome_flags", {})
    return list(flags.get(str(symbol), []))


def live_win_rate(orch, symbol: str, *, window: int = 12) -> float | None:
    """Taxa de acerto recente em trades reais do simbolo."""
    history = _symbol_history(orch, symbol)
    if len(history) < 4:
        return None
    tail = history[-min(window, len(history)) :]
    return sum(1 for x in tail if x) / float(len(tail))


def blended_val_accuracy(
    orch,
    symbol: str,
    val_accuracy: float,
    *,
    live_weight: float = 0.55,
    min_live_samples: int = 4,
) -> float:
    """Combina val_acc offline com win rate live (pessimista para gating)."""
    history = _symbol_history(orch, symbol)
    if len(history) < int(min_live_samples):
        return float(val_accuracy)
    live = live_win_rate(orch, symbol, window=max(int(min_live_samples), 4))
    weight = max(0.0, min(1.0, float(live_weight)))
    losses = sum(1 for x in history[-5:] if not x)
    if losses >= 2:
        weight = min(weight, 0.25)
    if losses >= 5:
        weight = min(weight, 0.45)
    blended = (1.0 - weight) * float(val_accuracy) + weight * float(live)
    out = min(float(val_accuracy), blended)
    if losses >= 5 and live is not None:
        out = min(out, float(val_accuracy) * 0.65)
    return out


def resolve_session_pause_config(dl_cfg: dict | None = None) -> dict:
    """Le knobs de pausa de sessao do SSOT deep_learning."""
    raw = merge_settings_block(("deep_learning",), dl_cfg if isinstance(dl_cfg, dict) else None)
    require_keys(raw, _SESSION_PAUSE_KEYS, "deep_learning")
    return {key: require_int(raw, key) for key in _SESSION_PAUSE_KEYS}


def tick_dl_session_pauses(orch) -> None:
    """Decrementa pausas de sessao por simbolo e limpa expiradas."""
    pauses = getattr(orch, "_dl_session_pause", None)
    until_map = getattr(orch, "_dl_session_pause_until", None)
    now = time.time()
    if isinstance(until_map, dict):
        for sym, deadline in list(until_map.items()):
            if float(deadline or 0.0) <= now:
                until_map.pop(sym, None)
                if isinstance(pauses, dict):
                    pauses.pop(str(sym), None)
    if not isinstance(pauses, dict):
        return
    dead: list[str] = []
    for sym, remaining in list(pauses.items()):
        nxt = int(remaining or 0) - 1
        if nxt <= 0:
            dead.append(str(sym))
        else:
            pauses[str(sym)] = nxt
    for sym in dead:
        pauses.pop(sym, None)
        if isinstance(until_map, dict):
            until_map.pop(sym, None)


def is_symbol_session_paused(orch, symbol: str) -> bool:
    """Indica pausa tecnica apos sequencia de losses no simbolo."""
    until_map = getattr(orch, "_dl_session_pause_until", None)
    if isinstance(until_map, dict):
        until = float(until_map.get(str(symbol), 0.0) or 0.0)
        if until > time.time():
            return True
    pauses = getattr(orch, "_dl_session_pause", None)
    if not isinstance(pauses, dict):
        return False
    return int(pauses.get(str(symbol), 0) or 0) > 0


def _session_cycle_seconds(orch) -> float:
    """Duracao de um ciclo M5 para converter pause_cycles em parede."""
    cfg = getattr(orch, "config", None)
    if isinstance(cfg, dict):
        orch_cfg = cfg.get("orchestrator")
        if isinstance(orch_cfg, dict):
            raw = orch_cfg.get("cycle_interval_seconds") or orch_cfg.get("signature_boundary_seconds") or 300
            try:
                return max(1.0, float(raw))
            except (TypeError, ValueError):
                return 300.0
    return 300.0


def maybe_pause_symbol_session(
    orch,
    symbol: str,
    *,
    max_losses_in_window: int,
    window_trades: int,
    pause_cycles: int,
) -> None:
    """Ativa pausa tecnica quando losses no tail atingem o limiar SSOT."""
    if int(pause_cycles) <= 0:
        return
    window = max(1, int(window_trades))
    need = max(1, int(max_losses_in_window))
    tail = _symbol_history(orch, symbol)[-window:]
    losses = sum(1 for won in tail if not won)
    if losses < need:
        return
    pauses = getattr(orch, "_dl_session_pause", None)
    if not isinstance(pauses, dict):
        pauses = {}
        orch._dl_session_pause = pauses
    pauses[str(symbol)] = int(pause_cycles)
    until_map = getattr(orch, "_dl_session_pause_until", None)
    if not isinstance(until_map, dict):
        until_map = {}
        orch._dl_session_pause_until = until_map
    deadline = time.time() + float(pause_cycles) * _session_cycle_seconds(orch)
    until_map[str(symbol)] = deadline
    prev = float(getattr(orch, "_cooldown_until", 0.0) or 0.0)
    orch._cooldown_until = max(prev, deadline)


def record_symbol_outcome(orch, symbol: str, *, won: bool, candle_epoch: int | None = None) -> None:
    """Registra resultado recente por simbolo para ponderar proximo treino."""
    if not hasattr(orch, "_dl_outcome_flags"):
        orch._dl_outcome_flags = {}
    if not hasattr(orch, "_dl_outcome_epochs"):
        orch._dl_outcome_epochs = {}
    sym = str(symbol)
    history = orch._dl_outcome_flags.setdefault(sym, [])
    history.append(bool(won))
    if len(history) > 80:
        del history[: len(history) - 80]
    if candle_epoch is not None:
        epochs = orch._dl_outcome_epochs.setdefault(sym, [])
        epochs.append(int(candle_epoch))
        if len(epochs) > 80:
            del epochs[: len(epochs) - 80]
    if not won:
        cfg = getattr(orch, "config", None)
        dl_cfg = cfg.get("deep_learning") if isinstance(cfg, dict) else None
        pause = resolve_session_pause_config(dl_cfg if isinstance(dl_cfg, dict) else None)
        maybe_pause_symbol_session(
            orch,
            sym,
            max_losses_in_window=int(pause["session_max_losses_in_window"]),
            window_trades=int(pause["session_window_trades"]),
            pause_cycles=int(pause["session_pause_cycles"]),
        )


def sample_weights_for_symbol(
    orch,
    symbol: str,
    sample_count: int,
    targets: list[float] | None = None,
) -> list[float]:
    """Gera pesos alinhados ao tail de treino com boost apos losses recentes."""
    if sample_count <= 0:
        return []
    flags = getattr(orch, "_dl_outcome_flags", {}).get(str(symbol), [])
    weights = [1.0] * sample_count
    if not flags:
        return weights
    tail = flags[-min(16, len(flags)) :]
    loss_ratio = 1.0 - (sum(1 for x in tail if x) / float(len(tail)))
    boost = 1.0 + loss_ratio * 1.35
    focus = min(sample_count, max(4, len(tail) * 3))
    for idx in range(sample_count - focus, sample_count):
        weights[idx] = boost
    wins = sum(1 for x in tail if x)
    if wins >= len(tail) - 1 and len(tail) >= 4:
        dampen = max(0.75, 1.0 - wins / float(len(tail) + 1))
        for idx in range(sample_count - focus, sample_count):
            weights[idx] *= dampen
    last_dir = getattr(orch, "_last_loss_direction", None)
    if last_dir and targets and len(targets) == sample_count and tail and not tail[-1]:
        want_label = 1.0 if str(last_dir).upper() == "CALL" else 0.0
        for idx in range(sample_count):
            if float(targets[idx]) == want_label:
                weights[idx] *= 1.25
            else:
                weights[idx] *= 1.55
    return weights
