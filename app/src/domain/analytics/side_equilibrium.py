"""Equilibrio CALL/PUT sob leis dos pequenos e grandes numeros."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any


ACTION_PASS = "pass"
ACTION_HARD_SKIP = "hard_skip"
ACTION_SOFT = "soft_penalty"


@dataclass(frozen=True)
class SideCounts:
    """Contagens rolling de CALL/PUT e utilitarios de WR/frequencia."""

    call_n: int = 0
    call_wins: int = 0
    put_n: int = 0
    put_wins: int = 0

    @property
    def total(self) -> int:
        """Total de trades CALL+PUT na janela."""
        return int(self.call_n) + int(self.put_n)

    def wr(self, side: str) -> float | None:
        """Win rate do lado solicitado ou None se sem amostras."""
        name = str(side).upper()
        if name == "CALL":
            if self.call_n <= 0:
                return None
            return float(self.call_wins) / float(self.call_n)
        if name == "PUT":
            if self.put_n <= 0:
                return None
            return float(self.put_wins) / float(self.put_n)
        return None

    def side_n(self, side: str) -> int:
        """Numero de trades do lado solicitado."""
        name = str(side).upper()
        if name == "CALL":
            return int(self.call_n)
        if name == "PUT":
            return int(self.put_n)
        return 0

    def freq_share(self, side: str) -> float:
        """Fracao de frequencia do lado no total da janela."""
        total = self.total
        if total <= 0:
            return 0.5
        return float(self.side_n(side)) / float(total)


@dataclass(frozen=True)
class SideEquilibriumConfig:
    """Parametros de equilibrio small-N e large-N."""

    enabled: bool = True
    small_window: int = 12
    large_window: int = 100
    n_min_small: int = 6
    n_min_large: int = 40
    wr_floor_small: float = 0.40
    wr_floor_large: float = 0.48
    freq_bias_max_small: float = 0.75
    freq_bias_max_large: float = 0.65
    break_even_wr: float = 0.55
    kelly_mult_soft: float = 0.55
    margin_boost_soft: float = 0.03


@dataclass(frozen=True)
class SideEquilibriumDecision:
    """Decisao de gate de equilibrio CALL/PUT."""

    action: str
    reason: str
    kelly_mult: float = 1.0
    margin_boost: float = 0.0
    call_n: int = 0
    call_wins: int = 0
    put_n: int = 0
    put_wins: int = 0
    side_wr: float | None = None
    freq_bias: float = 0.5
    z_vs_half: float = 0.0


def parse_side_equilibrium_config(raw: dict[str, Any] | None) -> SideEquilibriumConfig:
    """Converte dict de config em SideEquilibriumConfig."""
    cfg = raw if isinstance(raw, dict) else {}
    return SideEquilibriumConfig(
        enabled=bool(cfg.get("enabled", True)),
        small_window=max(4, int(cfg.get("small_window", 12))),
        large_window=max(20, int(cfg.get("large_window", 100))),
        n_min_small=max(3, int(cfg.get("n_min_small", 6))),
        n_min_large=max(10, int(cfg.get("n_min_large", 40))),
        wr_floor_small=float(cfg.get("wr_floor_small", 0.40)),
        wr_floor_large=float(cfg.get("wr_floor_large", 0.48)),
        freq_bias_max_small=float(cfg.get("freq_bias_max_small", 0.75)),
        freq_bias_max_large=float(cfg.get("freq_bias_max_large", 0.65)),
        break_even_wr=float(cfg.get("break_even_wr", 0.55)),
        kelly_mult_soft=max(0.05, min(1.0, float(cfg.get("kelly_mult_soft", 0.55)))),
        margin_boost_soft=max(0.0, float(cfg.get("margin_boost_soft", 0.03))),
    )


def binomial_z_vs_p(wins: int, n: int, p0: float = 0.5) -> float:
    """Z-score binomial do WR observado contra p0."""
    if n <= 0:
        return 0.0
    p_hat = float(wins) / float(n)
    p = min(max(float(p0), 1e-6), 1.0 - 1e-6)
    se = sqrt(p * (1.0 - p) / float(n))
    if se <= 1e-12:
        return 0.0
    return (p_hat - p) / se


def _pack_decision(
    counts: SideCounts,
    *,
    action: str,
    reason: str,
    wr: float | None,
    freq: float,
    z_half: float,
    kelly_mult: float = 1.0,
    margin_boost: float = 0.0,
) -> SideEquilibriumDecision:
    """Empacota SideEquilibriumDecision a partir das contagens."""
    return SideEquilibriumDecision(
        action=action,
        reason=reason,
        kelly_mult=kelly_mult,
        margin_boost=margin_boost,
        call_n=counts.call_n,
        call_wins=counts.call_wins,
        put_n=counts.put_n,
        put_wins=counts.put_wins,
        side_wr=wr,
        freq_bias=freq,
        z_vs_half=z_half,
    )


def _evaluate_small_regime(
    counts: SideCounts,
    *,
    config: SideEquilibriumConfig,
    wr: float | None,
    n_side: int,
    freq: float,
    z_half: float,
    base: SideEquilibriumDecision,
) -> SideEquilibriumDecision:
    """Aplica regras hard-skip do regime small-N."""
    if counts.total < config.n_min_small or n_side < max(3, config.n_min_small // 2):
        return _pack_decision(
            counts, action=ACTION_PASS, reason="small_n_insufficient", wr=wr, freq=freq, z_half=z_half
        )
    bad_wr = wr is not None and wr + 1e-12 < config.wr_floor_small
    hot_freq = freq + 1e-12 >= config.freq_bias_max_small
    if not (bad_wr or (hot_freq and (wr is None or wr + 1e-12 < config.break_even_wr))):
        return base
    reason = "side_imbalance_small_n"
    if hot_freq and bad_wr:
        reason = "side_imbalance_small_n_freq_wr"
    elif hot_freq:
        reason = "side_imbalance_small_n_freq"
    return _pack_decision(counts, action=ACTION_HARD_SKIP, reason=reason, wr=wr, freq=freq, z_half=z_half)


def _evaluate_large_regime(
    counts: SideCounts,
    *,
    config: SideEquilibriumConfig,
    wr: float | None,
    n_side: int,
    freq: float,
    z_half: float,
    base: SideEquilibriumDecision,
) -> SideEquilibriumDecision:
    """Aplica regras soft do regime large-N."""
    if counts.total < config.n_min_large or n_side < max(8, config.n_min_large // 3):
        return _pack_decision(
            counts, action=ACTION_PASS, reason="large_n_insufficient", wr=wr, freq=freq, z_half=z_half
        )
    bad_wr = wr is not None and wr + 1e-12 < config.wr_floor_large
    hot_freq = freq + 1e-12 >= config.freq_bias_max_large
    below_be = wr is not None and wr + 1e-12 < config.break_even_wr
    if not (bad_wr or (hot_freq and below_be)):
        return base
    return _pack_decision(
        counts,
        action=ACTION_SOFT,
        reason="side_imbalance_large_n",
        wr=wr,
        freq=freq,
        z_half=z_half,
        kelly_mult=config.kelly_mult_soft,
        margin_boost=config.margin_boost_soft,
    )


def evaluate_side_equilibrium(
    counts: SideCounts, proposed_side: str, *, config: SideEquilibriumConfig, regime: str
) -> SideEquilibriumDecision:
    """Avalia equilibrio CALL/PUT para o lado proposto."""
    side = str(proposed_side).upper()
    wr = counts.wr(side)
    n_side = counts.side_n(side)
    freq = counts.freq_share(side)
    wins = counts.call_wins if side == "CALL" else counts.put_wins
    z_half = binomial_z_vs_p(wins, n_side, 0.5) if n_side > 0 else 0.0
    base = _pack_decision(counts, action=ACTION_PASS, reason="ok", wr=wr, freq=freq, z_half=z_half)
    if not config.enabled:
        return _pack_decision(counts, action=ACTION_PASS, reason="disabled", wr=wr, freq=freq, z_half=z_half)
    if side not in {"CALL", "PUT"}:
        return base
    if regime == "small":
        return _evaluate_small_regime(counts, config=config, wr=wr, n_side=n_side, freq=freq, z_half=z_half, base=base)
    return _evaluate_large_regime(counts, config=config, wr=wr, n_side=n_side, freq=freq, z_half=z_half, base=base)
