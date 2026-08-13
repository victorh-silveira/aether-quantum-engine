"""Score e elegibilidade do sweep multi-TF (acima do breakeven)."""

from __future__ import annotations

from typing import Any

from src.application.services.deep_learning.tf_sweep_config import load_tf_sweep_knobs


def implied_breakeven(payout: float) -> float:
    """Breakeven WR para binaria com payout liquido b (ex.: 0.72 → ~0.581)."""
    b = max(0.0, float(payout))
    return 1.0 / (1.0 + b) if b > 0.0 else 1.0


def resolve_rank_wr(row: dict[str, Any]) -> float:
    """WR de ranking: settlement preferencial; fallback label so se settle ausente."""
    settle = row.get("settle_wr")
    if settle is None:
        settle = row.get("deploy_settlement_win_rate")
    if settle is not None:
        return float(settle)
    return float(row.get("val_accuracy") or 0.0)


def resolve_settle_n(row: dict[str, Any]) -> int:
    """N do mini-settlement; ausente = 0 (fail-closed)."""
    raw = row.get("settle_n")
    if raw is None:
        raw = row.get("deploy_settlement_n")
    try:
        return max(0, int(raw or 0))
    except (TypeError, ValueError):
        return 0


def is_tf_eligible(
    *,
    rank_wr: float,
    be_implied: float,
    min_edge_vs_breakeven: float,
    settle_n: int = 0,
    min_settle_n: int = 16,
    history_bars: int = 0,
    min_history_bars: int = 0,
    deploy_ok: bool | None = None,
    val_accuracy: float | None = None,
) -> bool:
    """Elegivel: settle_wr >= be+margem e N/historico acima do piso."""
    _ = (deploy_ok, val_accuracy)
    if int(settle_n) < int(min_settle_n):
        return False
    if int(min_history_bars) > 0 and int(history_bars) < int(min_history_bars):
        return False
    return float(rank_wr) + 1e-12 >= float(be_implied) + float(min_edge_vs_breakeven)


def _history_bars_for_settle(payload: dict[str, Any], settings: dict[str, Any] | None) -> int:
    """Historico efetivo do ckpt ou fallback SSOT (training/micro/history_bars)."""
    history = int(payload.get("training_history_bars") or payload.get("history_bars") or 0)
    if history > 0:
        return history
    if not isinstance(settings, dict):
        return 0
    dl = settings.get("deep_learning") if isinstance(settings.get("deep_learning"), dict) else {}
    data = settings.get("data_handler") if isinstance(settings.get("data_handler"), dict) else {}
    if isinstance(dl, dict) and dl.get("training_history_bars") is not None:
        return max(0, int(dl["training_history_bars"]))
    if isinstance(data, dict) and data.get("micro_history_bars") is not None:
        return max(0, int(data["micro_history_bars"]))
    if isinstance(data, dict) and data.get("history_bars") is not None:
        return max(0, int(data["history_bars"]))
    return 0


def checkpoint_settle_eligible(
    payload: dict[str, Any] | None,
    settings: dict[str, Any] | None,
    *,
    knobs: dict[str, Any] | None = None,
) -> bool:
    """True se o ckpt passa o mesmo criterio settle do sweep (ignora label ACC)."""
    if not isinstance(payload, dict) or not isinstance(settings, dict):
        return False
    sweep = knobs if isinstance(knobs, dict) else load_tf_sweep_knobs(settings)
    settle = payload.get("deploy_settlement_win_rate")
    if settle is None:
        return False
    be = implied_breakeven(float(sweep["payout_for_breakeven"]))
    return is_tf_eligible(
        rank_wr=float(settle),
        be_implied=be,
        min_edge_vs_breakeven=float(sweep["min_edge_vs_breakeven"]),
        settle_n=int(payload.get("deploy_settlement_n") or 0),
        min_settle_n=int(sweep.get("min_settle_n", 16)),
        history_bars=_history_bars_for_settle(payload, settings),
        min_history_bars=int(sweep.get("min_history_bars", 0)),
    )


def score_tf_row(
    row: dict[str, Any],
    *,
    knobs: dict[str, Any],
) -> float:
    """Score: edge de settlement, Brier, sharpness, meta IR."""
    be = float(row.get("be_implied") or implied_breakeven(float(knobs["payout_for_breakeven"])))
    wr = resolve_rank_wr(row)
    edge = wr - be
    brier = float(row.get("val_brier") or 0.0)
    soft_max = max(1e-6, float(knobs.get("soft_max_brier", 0.26)))
    brier_term = max(0.0, 1.0 - min(1.0, brier / soft_max))
    sharp = max(0.0, float(row.get("oos_sharpness") or 0.0))
    meta_ir = max(0.0, float(row.get("meta_ir") or 0.0))
    return (
        float(knobs["weight_edge"]) * edge
        + float(knobs["weight_brier"]) * brier_term
        + float(knobs["weight_sharpness"]) * sharp
        + float(knobs["weight_meta_ir"]) * meta_ir
    )


def enrich_leaderboard_row(
    row: dict[str, Any],
    *,
    knobs: dict[str, Any],
) -> dict[str, Any]:
    """Preenche be, edge_vs_be (settlement), eligible e score."""
    out = dict(row)
    be = implied_breakeven(float(knobs["payout_for_breakeven"]))
    wr = resolve_rank_wr(out)
    settle_n = resolve_settle_n(out)
    history = int(out.get("history_bars") or out.get("training_history_bars") or 0)
    out["be_implied"] = round(be, 6)
    out["rank_wr"] = round(wr, 6)
    out["settle_n"] = settle_n
    out["edge_vs_be"] = round(wr - be, 6)
    out["eligible"] = is_tf_eligible(
        rank_wr=wr,
        be_implied=be,
        min_edge_vs_breakeven=float(knobs["min_edge_vs_breakeven"]),
        settle_n=settle_n,
        min_settle_n=int(knobs.get("min_settle_n", 16)),
        history_bars=history,
        min_history_bars=int(knobs.get("min_history_bars", 0)),
        deploy_ok=bool(out.get("deploy_ok")),
        val_accuracy=float(out.get("val_accuracy") or 0.0),
    )
    out["score"] = round(score_tf_row(out, knobs=knobs), 8) if out["eligible"] else None
    return out


def pick_tf_winner(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Argmax score entre elegiveis; empate: maior edge_vs_be, menor Brier."""
    eligible = [r for r in rows if isinstance(r, dict) and bool(r.get("eligible"))]
    if not eligible:
        return None

    def _key(row: dict[str, Any]) -> tuple[float, float, float, int]:
        """Chave de desempate: score, edge_vs_be, -brier, settle_n."""
        score = float(row.get("score") or 0.0)
        edge = float(row.get("edge_vs_be") or 0.0)
        brier = float(row.get("val_brier") or 1.0)
        settle_n = resolve_settle_n(row)
        return (score, edge, -brier, settle_n)

    return max(eligible, key=_key)
