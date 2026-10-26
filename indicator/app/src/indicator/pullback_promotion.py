"""Ranqueamento diagnostico do proxy de velas, sem promocao operacional."""

from __future__ import annotations

from typing import Any


def rank_m5_pairs(
    report: dict[str, dict],
    excluded_symbols: set[str] | None = None,
    payout: float = 0.85,
    target_period: int = 900,
) -> list[dict[str, Any]]:
    """Ranqueia pares com base em EV, taxa de acerto e consistencia."""
    excluded = excluded_symbols or set()
    ranked = []
    suffix = f":{target_period}"
    for key, data in report.items():
        if not key.endswith(suffix):
            continue
        symbol = key.split(":")[0]
        if data.get("estado") != "PROXY DE VELAS":
            continue
        setups = int(data.get("setups", 0))
        wins = int(data.get("acertos", 0))
        losses = int(data.get("erros", 0))
        ties = int(data.get("empates", 0))
        max_consecutive_losses = int(data.get("max_losses_consecutivos", 0))
        decisions = wins + losses
        win_rate = (wins / decisions) if decisions > 0 else 0.0
        ev = (win_rate * payout) - ((1.0 - win_rate) * 1.0) if decisions > 0 else -1.0
        is_eligible = symbol not in excluded and setups > 0
        ranked.append(
            {
                "symbol": symbol,
                "period": target_period,
                "setups": setups,
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "max_consecutive_losses": max_consecutive_losses,
                "decisions": decisions,
                "win_rate": win_rate,
                "ev": ev,
                "eligible": is_eligible,
                "excluded": symbol in excluded,
                "optimal_hours_brt": data.get("optimal_hours_brt", []),
                "best_window_brt": data.get("best_window_brt", "05:00 as 13:00 (BRT)"),
                "hourly_profile_brt": data.get("hourly_profile_brt", {}),
            }
        )

    def _sort_key(item: dict[str, Any]) -> tuple[int, float, float, int, int]:
        elig = 1 if item["eligible"] else 0
        max_loss = int(item.get("max_consecutive_losses", 999))
        return (elig, item["ev"], item["win_rate"], -max_loss, item["setups"])

    ranked.sort(key=_sort_key, reverse=True)
    return ranked


def format_ranking_table(ranked: list[dict[str, Any]]) -> str:
    """Gera tabela visual do ranking dos pares M15/H1/D1 avaliados."""
    target_period = int(ranked[0].get("period", 900)) if ranked else 900
    tf_name = (
        "D1"
        if target_period == 86400
        else (f"H{target_period // 3600}" if target_period >= 3600 else f"M{target_period // 60}")
    )
    lines = [
        "=" * 100,
        f"RANKING DO SETUP PULLBACK {tf_name} (PROXY DE VELAS)",
        "=" * 100,
        f"{'POS':<4} | {'ATIVO':<8} | {'SETUPS':<7} | {'WINS':<5} | {'LOSS':<5} | {'EMP':<4} | {'MAX LOSS':<8} | {'ACERTO %':<9} | {'EV (85%)':<9} | {'STATUS':<12}",
        "-" * 100,
    ]
    for idx, row in enumerate(ranked, 1):
        status = "EXCLUIDO" if row["excluded"] else ("LIDER PROXY" if idx == 1 and row["eligible"] else "CANDIDATO")
        wr_pct = f"{row['win_rate'] * 100:.1f}%"
        ev_str = f"{row['ev']:+.4f}"
        max_l = str(row.get("max_consecutive_losses", 0))
        lines.append(
            f"{idx:<4} | {row['symbol']:<8} | {row['setups']:<7} | {row['wins']:<5} | "
            f"{row['losses']:<5} | {row['ties']:<4} | {max_l:<8} | {wr_pct:<9} | {ev_str:<9} | {status:<12}"
        )
    lines.append("=" * 100)
    return "\n".join(lines)
