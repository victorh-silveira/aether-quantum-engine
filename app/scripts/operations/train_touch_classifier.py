"""Treina probabilidade de toque com propostas observadas e ticks, sem enviar ordens."""

import argparse
import asyncio
import json
import sys
from pathlib import Path


_APP = Path(__file__).resolve().parents[2]
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

from src.application.services.touch_training import build_touch_samples, train_touch_model
from src.domain.config_knobs import load_settings_json
from src.domain.models.touch_policy import resolve_touch_policy, touch_enabled
from src.infrastructure.market.touch_store import read_touch_quotes, read_touch_ticks, save_touch_model


async def main() -> int:
    """Codigo 3: modo legado; 2: dados invalidos/ausentes; 0: candidato avaliado, nao necessariamente aprovado."""
    parser = argparse.ArgumentParser(description="Treino Touch/No Touch 1HZ75V")
    parser.add_argument("--if-enabled", action="store_true")
    args = parser.parse_args()
    settings = load_settings_json()
    if args.if_enabled and not touch_enabled(settings):
        return 3
    try:
        policy = resolve_touch_policy(settings)
        try:
            quotes = read_touch_quotes()
        except FileNotFoundError:
            raise FileNotFoundError(
                "data/touch/quotes.jsonl ausente. Execute 'python app/scripts/operations/collect_touch_dataset.py' "
                "para coletar cotacoes e ticks antes do treino Touch (ou defina touch.enabled=false em "
                "config/settings.json para rodar o pipeline TCN legado)"
            ) from None
        if not quotes:
            raise ValueError(
                "Journal sem propostas em data/touch/quotes.jsonl; colete cotacoes e ticks via "
                "'python app/scripts/operations/collect_touch_dataset.py' antes do treino"
            )
        start = min(q["group_ms"] for q in quotes) - 2000
        end = max(q["decision_ms"] for q in quotes) + policy.duration_seconds * 1000 + max(policy.latency_ms) + 2000
        ticks = await read_touch_ticks(settings["infra"]["timescale"]["dsn"], policy.symbol, start, end)
        rows, rejected = await asyncio.to_thread(build_touch_samples, quotes, ticks, policy)
        bundle = await asyncio.to_thread(train_touch_model, rows, policy)
        bundle["rejected_quotes"] = rejected
        path = save_touch_model(bundle)
        print(f"TOUCH | artifact={path} qualified={bundle['qualified']} rejected={rejected}")
        print(json.dumps(bundle["oos"], allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"TOUCH | treino nao concluido: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
