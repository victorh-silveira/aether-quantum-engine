"""Comandos offline: prepare, train, evaluate e promote, sem executor de ordens."""

import argparse
import asyncio
import json
import time
from pathlib import Path

from tools.laya_workflow.artifacts import read_json
from tools.laya_workflow.dataset import prepare
from tools.laya_workflow.evaluation import evaluate, promote
from tools.laya_workflow.training import train


def main() -> None:
    """Executa uma etapa explicita sem ativar modelos automaticamente."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "evaluate", "promote"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, default=Path(".cache/laya-r10-gpu-run-fast"))
    parser.add_argument("--settings", type=Path, default=Path("config/settings.json"))
    parser.add_argument("--cutoff", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--profile", choices=("auto", "head", "full"), default="auto")
    parser.add_argument("--preset", choices=("fast", "complete"), default="fast")
    parser.add_argument("--warm-start", type=Path, default=None)
    parser.add_argument("--warm-start-auto", type=Path, default=None)
    parser.add_argument("--completed-epochs", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-unapproved-override", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        result = asyncio.run(prepare(args.data, read_json(args.settings), args.cutoff or int(time.time())))
    elif args.command == "train":
        warm = args.warm_start
        if warm is None and args.preset == "fast" and args.warm_start_auto and args.warm_start_auto.exists():
            warm = args.warm_start_auto
        result = train(
            args.data,
            args.run,
            dry_run=args.dry_run,
            profile=args.profile,
            preset=args.preset,
            warm_start=warm,
            completed_epochs=args.completed_epochs if warm else 0,
            resume=args.resume,
        )
    elif args.command == "evaluate":
        result = evaluate(args.data, args.run, args.device)
    else:
        result = promote(args.data, args.run, allow_unapproved_override=args.allow_unapproved_override)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
