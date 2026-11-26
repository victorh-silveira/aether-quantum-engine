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
    parser.add_argument("--run", type=Path, default=Path(".cache/laya-gpu-run"))
    parser.add_argument("--settings", type=Path, default=Path("config/settings.json"))
    parser.add_argument("--cutoff", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.command == "prepare":
        result = asyncio.run(prepare(args.data, read_json(args.settings), args.cutoff or int(time.time())))
    elif args.command == "train":
        result = train(args.data, args.run, dry_run=args.dry_run)
    elif args.command == "evaluate":
        result = evaluate(args.data, args.run, args.device)
    else:
        result = promote(args.data, args.run)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
