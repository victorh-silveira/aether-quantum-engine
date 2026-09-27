"""Treino temporal Touch e avaliacao da politica completa em propostas OOS."""

import time

import numpy as np
from scipy.stats import t
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.application.services.touch_model import predict_touch, select_touch_quote
from src.domain.math.touch_ticks import replay_touch_labels, tick_arrays
from src.domain.models.touch_policy import TouchPolicy


def build_touch_samples(quotes: list[dict], ticks, policy: TouchPolicy) -> tuple[list[dict], int]:
    """Mantem cotacoes reais compativeis e trajetorias completas; contabiliza rejeicoes."""
    rows, rejected, seen = [], 0, set()
    epochs, prices = tick_arrays(ticks)
    for quote in sorted(quotes, key=lambda row: row["decision_ms"]):
        key = str(quote["proposal_id"])
        if key in seen:
            continue
        seen.add(key)
        try:
            if quote["policy_hash"] != policy.fingerprint() or quote["symbol"] != policy.symbol:
                raise ValueError("Proposta de outro contrato/schema")
            start = np.searchsorted(epochs, quote["decision_ms"]) - 1
            end = (
                np.searchsorted(
                    epochs, quote["decision_ms"] + policy.duration_seconds * 1000 + max(policy.latency_ms), side="right"
                )
                + 1
            )
            labels = replay_touch_labels(
                np.column_stack((epochs[max(0, start) : end], prices[max(0, start) : end])), quote, policy
            )
            rows.append({**quote, "labels": labels})
        except (ValueError, KeyError, TypeError):
            rejected += 1
    return rows, rejected


def split_touch_samples(rows: list[dict], policy: TouchPolicy) -> tuple[list, list, list]:
    """Separa grupos cronologicos, purgando horizonte e janela de features entre conjuntos."""
    groups = sorted({row["group_ms"] for row in rows})
    if len(groups) < policy.min_train_groups:
        raise ValueError(f"Touch: dados insuficientes ({len(groups)}/{policy.min_train_groups} grupos)")
    cut1, cut2 = groups[int(len(groups) * 0.6)], groups[int(len(groups) * 0.8)]
    embargo = (policy.duration_seconds + policy.history_seconds) * 1000 + max(policy.latency_ms)
    train = [r for r in rows if r["decision_ms"] + embargo < cut1]
    calibration = [r for r in rows if cut1 <= r["group_ms"] and r["decision_ms"] + embargo < cut2]
    test = [r for r in rows if r["group_ms"] >= cut2]
    if min(len(train), len(calibration), len(test)) < 20:
        raise ValueError("Touch: folds insuficientes apos purging")
    return train, calibration, test


def _unique_targets(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Touch e No Touch da mesma barreira nao duplicam a amostra de treino."""
    unique = {(r["group_ms"], r["barrier"]): r for r in rows}
    x = np.asarray([r["features"] for r in unique.values()], dtype=float)
    y = np.asarray([r["labels"][-1] for r in unique.values()], dtype=int)
    if x.ndim != 2 or x.shape[1] != 5 or not np.isfinite(x).all() or len(np.unique(y)) != 2:
        raise ValueError("Touch: features invalidas ou fold com classe unica")
    return x, y


def evaluate_touch_policy(bundle: dict, rows: list[dict], policy: TouchPolicy) -> dict:
    """Uma posicao por janela, payout observado e pior retorno nos cenarios de latencia."""
    groups = {}
    for row in rows:
        groups.setdefault(row["group_ms"], []).append(row)
    returns, briers, available_at = [], [], 0
    for epoch in sorted(groups):
        if epoch <= available_at:
            continue
        selection = select_touch_quote(bundle, groups[epoch], policy)
        if selection is None:
            continue
        quote, prob, _ = selection
        wins = quote["labels"] if quote["contract_type"] == "ONETOUCH" else [1 - y for y in quote["labels"]]
        returns.append(min(wins) * quote["payout"] / quote["ask_price"] - 1)
        briers.append((prob - quote["labels"][-1]) ** 2)
        available_at = quote["decision_ms"] + policy.duration_seconds * 1000 + max(policy.latency_ms)
    n = len(returns)
    mean = float(np.mean(returns)) if n else 0.0
    lcb = mean - float(t.ppf(0.90, n - 1)) * float(np.std(returns, ddof=1)) / np.sqrt(n) if n > 1 else -1.0
    return {
        "source": "quoted_tick_replay",
        "trades": n,
        "mean_return": mean,
        "return_lcb90": float(lcb),
        "brier": float(np.mean(briers)) if n else 1.0,
    }


def train_touch_model(rows: list[dict], policy: TouchPolicy) -> dict:
    """Ajusta modelo e calibrador antes de consultar teste; nao otimiza thresholds no OOS."""
    train, calibration, test = split_touch_samples(rows, policy)
    x, y = _unique_targets(train)
    xc, yc = _unique_targets(calibration)
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=0.1, max_iter=500, random_state=42, solver="liblinear").fit(scaler.transform(x), y)
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(model.predict_proba(scaler.transform(xc))[:, 1], yc)
    bundle = {
        "schema": "touch_ticks_v1",
        "policy_hash": policy.fingerprint(),
        "trained_ms": int(time.time() * 1000),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "cal_x": calibrator.X_thresholds_.tolist(),
        "cal_y": calibrator.y_thresholds_.tolist(),
        "qualified": False,
        "train_end_ms": max(r["decision_ms"] for r in train),
        "calibration_start_ms": min(r["decision_ms"] for r in calibration),
        "test_start_ms": min(r["decision_ms"] for r in test),
    }
    xt, yt = _unique_targets(test)
    predictions = np.array([predict_touch(bundle, row.tolist()) for row in xt])
    brier = float(np.mean((predictions - yt) ** 2))
    baseline_brier = float(np.mean((float(np.mean(yc)) - yt) ** 2))
    report = evaluate_touch_policy(bundle, test, policy)
    report.update({"all_brier": brier, "baseline_brier": baseline_brier})
    bundle["oos"] = report
    bundle["qualified"] = bool(
        report["trades"] >= policy.min_oos_trades and report["return_lcb90"] > 0 and brier < baseline_brier
    )
    return bundle
