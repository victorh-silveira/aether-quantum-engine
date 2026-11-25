"""Pesquisa congelada de frequencia em OHLC, sem promover configuracao operacional."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import polars as pl

from scripts.analysis.frequency_replay import Signal, replay_portfolio
from scripts.analysis.frequency_signals import comparator_signals, persistent_signals, read_dataset, session_gap_signals
from scripts.analysis.frequency_stats import BRT, PAYOUTS, daily_counts, holm_adjust, infer_ev, select_candidate
from src.domain.fundamental.economic_calendar import policy_from_config
from src.indicator.frequency_candidate import Candidate, generate_candidates
from src.infrastructure.market.economic_calendar_client import parse_calendar_payload


SCHEMA = {"symbol": pl.String, "entry_epoch": pl.Int64, "side": pl.String, "outcome": pl.Int64}


def _hash(path: Path) -> str:
    """Hash dos bytes para reuso de artefatos sem inferir integridade por nome."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, body: object) -> None:
    """Persiste JSON legivel, rejeitando NaN e infinito."""
    path.write_text(json.dumps(body, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def cache_signals(output: Path, manifest: dict, *, stage: str) -> tuple[dict, dict]:
    """Reutiliza sinais somente sob mesmos dados e mesmo codigo de pesquisa."""
    paths = [
        Path("scripts/analysis/frequency_features.py"),
        Path("scripts/analysis/frequency_signals.py"),
        Path("src/indicator/frequency_candidate.py"),
        Path("src/indicator/strategy_persistent_trend.py"),
        Path("src/indicator/pullback.py"),
        Path("src/indicator/pullback_dual_sniper.py"),
        Path("src/indicator/pullback_senior.py"),
        Path("src/indicator/strategy_multi_regime.py"),
        Path("src/indicator/pullback_indicators.py"),
        Path("src/indicator/institutional.py"),
        Path("src/indicator/pullback_trend_regime.py"),
    ]
    fingerprint = hashlib.sha256(
        (
            json.dumps({key: manifest[key] for key in ("sha256", "start_epoch", "cutoff_epoch")}, sort_keys=True)
            + "".join(_hash(path) for path in paths)
        ).encode()
    ).hexdigest()
    cache = output / "signals"
    cache.mkdir(exist_ok=True)
    meta_path = cache / "index.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    valid = meta.get("fingerprint") == fingerprint
    hashes = dict(meta.get("hashes", {})) if valid else {}
    for name, expected_hash in hashes.items():
        path = cache / f"{name}.parquet"
        if not path.is_file() or _hash(path) != expected_hash:
            raise ValueError(f"Hash de sinais divergente: {name}")
    labels = [c.label for c in generate_candidates()]
    groups = {"persistent": labels, "comparators": ["dual_sniper", "multi_regime_hf", "senior_institutional"]}
    signals = {}
    dataset = None
    for group, names in groups.items():
        if stage not in ("all", group):
            continue
        cached = valid and all(
            (cache / f"{name}.parquet").is_file() and name in meta.get("hashes", {}) for name in names
        )
        if cached:
            for name in names:
                path = cache / f"{name}.parquet"
                if meta.get("hashes", {}).get(name) != _hash(path):
                    raise ValueError(f"Hash de sinais divergente: {name}")
                signals[name] = [Signal(*row) for row in pl.read_parquet(path).iter_rows()]
            print(f"REUSO sinais {group}", flush=True)
            continue
        if dataset is None:
            dataset = read_dataset(output / "candles.parquet")
        if group == "persistent":
            generated, audit = persistent_signals(dataset, manifest["start_epoch"], manifest["cutoff_epoch"])
            _write(output / "feature_audit.json", audit)
        else:
            generated = comparator_signals(dataset, manifest["start_epoch"], manifest["cutoff_epoch"])
        for name, rows in generated.items():
            records = [{key: value for key, value in asdict(row).items() if key != "selection_score"} for row in rows]
            pl.DataFrame(records, schema=SCHEMA).write_parquet(cache / f"{name}.parquet")
        signals.update(generated)
    hashes.update({name: _hash(cache / f"{name}.parquet") for name in signals})
    _write(meta_path, {"fingerprint": fingerprint, "hashes": hashes, "code_hashes": {str(p): _hash(p) for p in paths}})
    return signals, {"fingerprint": fingerprint, "complete_family": len(signals) == 148}


def _epoch(day: str) -> int:
    """Interpreta fronteiras de relatorio em BRT, incluindo dias zerados."""
    return int(datetime.fromisoformat(day).replace(tzinfo=BRT).timestamp())


def evaluate_window(signals: dict, start: int, end: int, *, events: tuple, policy, samples: int) -> tuple[list, dict]:
    """Reinicia banca por hipotese e janela; testa toda a familia congelada."""
    first = datetime.fromtimestamp(start, BRT).date().isoformat()
    last = datetime.fromtimestamp(end, BRT).date().isoformat()
    rows, details = [], {}
    for label, entries in signals.items():
        replay = replay_portfolio(
            entries, payout=0.70, events=events, calendar_policy=policy, start_epoch=start, end_epoch=end
        )
        _, counts = daily_counts(replay["trades"], first, last)
        stats = infer_ev(counts, samples=samples)
        rows.append(
            {
                "label": label,
                **stats,
                "trades_per_day": replay["mean_daily_trades"],
                "max_drawdown": replay["max_drawdown_pct"],
                "zero_trade_days": replay["zero_trade_days"],
                "cluster_budget_breaches": replay["counts"].get("cluster_budget_breaches", 0),
            }
        )
        details[label] = replay
    return holm_adjust(rows), details


def run(output: Path, *, stage: str = "all", samples: int = 10000) -> dict:
    """Executa diagnostico retrospectivo e impede promocao sem calendario verificado."""
    manifest = json.loads((output / "manifest.json").read_text())
    if _hash(output / "candles.parquet") != manifest["sha256"]:
        raise ValueError("Dataset alterado apos congelamento")
    config = json.loads((output / "settings_baseline.json").read_text())
    policy = policy_from_config(config)
    snapshot = output / "economic_calendar_snapshot.json"
    raw = json.loads(snapshot.read_text()) if snapshot.is_file() else {}
    events = tuple(parse_calendar_payload(raw.get("events", [])))
    signals, identity = cache_signals(output, manifest, stage=stage)
    start, cutoff = max(manifest["start_epoch"], _epoch("2026-01-01")), manifest["cutoff_epoch"]
    oos_start = _epoch("2026-04-01")
    oos_end = min(_epoch("2026-11-01") - 1, cutoff - 900)
    rows, details = evaluate_window(signals, oos_start, oos_end, events=events, policy=policy, samples=samples)
    _write(output / "oos_results.json", rows)
    print("OOS: bootstrap e Holm concluidos", flush=True)
    baseline_label = Candidate().label
    baseline_rate = next((r["trades_per_day"] for r in rows if r["label"] == baseline_label), float("inf"))
    data_qualified = bool(
        manifest.get("qualifica_validacao_anual_completa")
        and manifest["calendar"]["coverage_complete"]
        and identity["complete_family"]
    )
    for row in rows:
        row["higher_frequency_than_baseline"] = row["trades_per_day"] > baseline_rate
        row["statistically_eligible"] &= row["cluster_budget_breaches"] == 0 and row["higher_frequency_than_baseline"]
    winner = select_candidate(rows, data_qualified=data_qualified)
    baselines = {}
    for name, (left, right) in {
        "year_to_date": (start, cutoff - 900),
        "november": (_epoch("2026-11-01"), cutoff - 900),
        "today_original_cutoff": (_epoch("2026-11-25"), min(_epoch("2026-11-25T16:00:00") - 900, cutoff - 900)),
        "today_latest": (_epoch("2026-11-25"), cutoff - 900),
    }.items():
        baseline = replay_portfolio(
            signals.get(baseline_label, []),
            payout=0.7235,
            events=events,
            calendar_policy=policy,
            start_epoch=left,
            end_epoch=right,
        )
        baselines[name] = baseline
    ranked = sorted(rows, key=lambda r: (r["trades_per_day"], r["lower95"] or -10), reverse=True)
    scenarios = {}
    for row in ranked:
        label = row["label"]
        scenarios[label] = []
        for payout in PAYOUTS:
            for bankroll in (1000.0, 7000.0):
                replay = replay_portfolio(
                    signals[label],
                    payout=payout,
                    initial_bankroll=bankroll,
                    events=events,
                    calendar_policy=policy,
                    start_epoch=oos_start,
                    end_epoch=oos_end,
                )
                scenarios[label].append({k: v for k, v in replay.items() if k not in ("trades", "cycles", "daily")})
    print("CENARIOS: payouts e bancas concluidos", flush=True)
    folds = []
    for month in range(4, 11):
        train_end = _epoch(f"2026-{month:02d}-01") - 1
        training, _ = evaluate_window(signals, start, train_end, events=events, policy=policy, samples=samples)
        for row in training:
            row["statistically_eligible"] &= row["cluster_budget_breaches"] == 0
        choice = select_candidate(training, data_qualified=True)
        test_end = _epoch(f"2026-{month + 1:02d}-01") - 1
        test = replay_portfolio(
            signals[choice["label"]] if choice else [],
            payout=0.70,
            events=events,
            calendar_policy=policy,
            start_epoch=train_end + 1,
            end_epoch=test_end,
        )
        print(f"WALK_FORWARD 2026-{month:02d} concluido", flush=True)
        folds.append(
            {
                "month": f"2026-{month:02d}",
                "training_end": train_end,
                "statistical_choice_only": choice,
                "data_qualified": data_qualified,
                "test": {k: v for k, v in test.items() if k not in ("trades", "cycles")},
            }
        )
    boundary_signals = session_gap_signals(read_dataset(output / "candles.parquet"), _epoch("2026-11-01"), cutoff)
    report = {
        "identity": identity,
        "research_code_hashes": {str(p): _hash(p) for p in Path("scripts/analysis").glob("frequency_*.py")},
        "manifest_sha256": _hash(output / "manifest.json"),
        "calendar_snapshot_sha256": _hash(snapshot) if snapshot.is_file() else None,
        "calendar_coverage_complete": manifest["calendar"]["coverage_complete"],
        "qualifies_contract": False,
        "data_qualified": data_qualified,
        "baseline": baselines,
        "baseline_reconciliation": {
            "provided_november": {"trades": 60, "wins": 39, "losses": 21},
            "strict_contiguous_trades": baselines["november"]["trade_count"],
            "excluded_session_gap_signals": boundary_signals,
            "note": "A producao tolera gaps de sessao OTC; pesquisa aplica mandato de M15 contiguo.",
        },
        "oos": ranked,
        "scenarios": scenarios,
        "walk_forward": folds,
        "winner": winner,
        "bootstrap_samples": samples,
        "seed": 42,
        "limitations": [
            "2026 e novembro incompletos; 2026 ja influenciou a escolha da v3.",
            "Proxy OHLC nao reconstroi payout, entrada, empate e liquidacao broker.",
            "Ranking FTSE/lexical e desempate deterministico substituem snapshots broker ausentes.",
            "Calendario semanal cobre somente sua semana; ausencia historica impede promocao.",
        ],
        "demo": {
            "enabled": False,
            "candidate": winner,
            "required_settled_contracts": 200,
            "reason": "Sem candidato plenamente qualificado" if winner is None else "Exige piloto prospectivo",
        },
    }
    _write(output / "report.json", report)
    _write(output / "demo_candidate.json", report["demo"])
    _write(
        output / "candidate_grid.json",
        [{**asdict(c), "label": c.label, "version": c.version} for c in generate_candidates()],
    )
    for label in {baseline_label, *(r["label"] for r in ranked[:5])} & details.keys():
        _write(output / f"replay_{label}.json", details[label])
    print(
        json.dumps({"hypotheses": len(rows), "winner": winner, "data_qualified": data_qualified}, ensure_ascii=False),
        flush=True,
    )
    return report


def main() -> None:
    """Executa pesquisa sem permitir alteracoes de parametros pela CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".cache/frequency-20261125"))
    parser.add_argument("--stage", choices=("all", "persistent", "comparators"), default="all")
    args = parser.parse_args()
    run(args.output, stage=args.stage)


if __name__ == "__main__":
    main()
