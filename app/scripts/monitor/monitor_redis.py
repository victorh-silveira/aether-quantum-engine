"""Leitura sincrona das chaves de sessao ativa no Redis."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from scripts.monitor.monitor_state import DashboardState

try:
    import redis
except ImportError:
    redis = None


def refresh_session_targets_from_redis(state: DashboardState) -> bool:
    if redis is None or not state.redis_url:
        return False
    prefix = str(state.redis_key_prefix or "aether").rstrip(":")
    start_key = f"{prefix}:session:current:start_balance"
    target_key = f"{prefix}:session:current:target_win"
    try:
        client = redis.from_url(state.redis_url, decode_responses=True, socket_connect_timeout=0.5)
        start_raw = client.get(start_key)
        target_raw = client.get(target_key)
        if start_raw:
            start_val = float(start_raw)
            if start_val > 0.0:
                state.session_start_balance = start_val
        if target_raw:
            target_val = float(target_raw)
            if target_val > 0.0:
                state.session_target_win = target_val
        return bool(start_raw and target_raw)
    except Exception:
        return False
