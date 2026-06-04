"""Valida PAT (Bearer) + Deriv-App-ID contra a API REST Deriv."""

import argparse
import asyncio
import os
import sys
from pathlib import Path


_APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_APP))

from dotenv import load_dotenv

from aether_paths import REPO_ROOT
from src.infrastructure.api.deriv_credentials import is_legacy_deriv_app_id, persist_deriv_app_id
from src.infrastructure.api.deriv_pat_binding import looks_like_pat, parse_deriv_pat
from src.infrastructure.api.deriv_pat_session import DerivPatSession
from src.infrastructure.api.deriv_rest_client import DerivRestError


async def _async_main(args: argparse.Namespace) -> int:
    load_dotenv(REPO_ROOT / ".env")
    raw_pat = (args.pat or os.getenv("AETHER_DERIV_PAT") or "").strip()
    if not raw_pat:
        print("PAT ausente. Use AETHER_DERIV_PAT no .env", file=sys.stderr)
        return 1
    pat, inline_app = parse_deriv_pat(raw_pat)
    app_id_arg = (args.app_id or "").strip()
    if app_id_arg.startswith("pat_"):
        if not pat:
            pat = app_id_arg
        app_id_arg = ""
    if inline_app and not app_id_arg:
        app_id_arg = inline_app
    account = os.getenv("AETHER_DERIV_ACCOUNT_ID") or os.getenv("AETHER_OAUTH_ACCOUNT_ID")
    session = DerivPatSession(pat, app_id=app_id_arg or None, account_id=account)
    try:
        result = await session.bootstrap(persist_binding=args.save_env)
    except DerivRestError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if app_id_arg and is_legacy_deriv_app_id(app_id_arg):
        print(f"App ID {app_id_arg} e legado.", file=sys.stderr)
        return 1
    if looks_like_pat(result.app_id):
        print("App ID invalido (nao use pat_...).", file=sys.stderr)
        return 1
    if args.save_env:
        persist_deriv_app_id(REPO_ROOT, result.app_id)
        print(f"Gravado AETHER_DERIV_APP_ID={result.app_id} no .env")
    print("OK: PAT + App ID validos.")
    print(f"  app_id={result.app_id} conta={result.account_id} saldo={result.balance}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Valida PAT + App ID na Deriv API")
    parser.add_argument("--app-id", default=None)
    parser.add_argument("--pat", default=None)
    parser.add_argument("--save-env", action="store_true")
    args = parser.parse_args()
    return asyncio.run(_async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
