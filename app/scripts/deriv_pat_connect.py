"""Fluxo completo Deriv com PAT: health, accounts, otp e ping WebSocket."""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path


_APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_APP))

from dotenv import load_dotenv

from aether_paths import REPO_ROOT
from src.infrastructure.api.deriv_pat_binding import (
    DerivPatBindingError,
    discover_app_id_for_pat,
    looks_like_pat,
    parse_deriv_pat,
    probe_accounts_ok,
    save_binding,
)
from src.infrastructure.api.deriv_pat_session import DerivPatSession
from src.infrastructure.api.deriv_rest_client import DerivRestError


def _print_app_id_help() -> None:
    print(
        "\nComo obter o App ID (uma vez):\n"
        "  1. Abra https://developers.deriv.com/playground\n"
        "  2. Cole a mesma PAT no campo de token do topo\n"
        "  3. Endpoint: GET /trading/v1/options/accounts > Send\n"
        "  4. Se status 200: F12 > Rede > request accounts > header Deriv-App-ID\n"
        "  5. Rode de novo este script e cole o valor quando pedido,\n"
        "     ou: python app/scripts/deriv_pat_connect.py --app-id VALOR --save-binding\n"
    )


def _prompt_app_id(pat: str) -> str | None:
    if not sys.stdin.isatty():
        return None
    _print_app_id_help()
    try:
        entered = input("Deriv-App-ID: ").strip()
    except EOFError:
        return None
    if not entered or looks_like_pat(entered):
        return None
    if not probe_accounts_ok(pat, entered, "https://api.derivws.com", 30.0):
        print("App ID rejeitado pela API (401). Confira se e o header do mesmo app da PAT.", file=sys.stderr)
        return None
    save_binding(REPO_ROOT, pat, entered)
    print("App ID validado e salvo em app/data/deriv/pat_bindings.json")
    return entered


def _resolve_app_id(pat: str, cli_app_id: str | None, *, allow_prompt: bool) -> str | None:
    if cli_app_id and cli_app_id.strip():
        return cli_app_id.strip()
    try:
        return discover_app_id_for_pat(pat, REPO_ROOT)
    except DerivPatBindingError:
        if allow_prompt:
            return _prompt_app_id(pat)
        return None


def _load_pat(cli_pat: str | None) -> str:
    raw = (cli_pat or os.getenv("AETHER_DERIV_PAT") or "").strip()
    if not raw:
        raise SystemExit("Defina AETHER_DERIV_PAT no .env ou use --pat")
    token, _ = parse_deriv_pat(raw)
    return token


async def _run(args: argparse.Namespace) -> int:
    load_dotenv(REPO_ROOT / ".env")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    pat = _load_pat(args.pat)
    account = (
        args.account or os.getenv("AETHER_DERIV_ACCOUNT_ID") or os.getenv("AETHER_OAUTH_ACCOUNT_ID") or ""
    ).strip() or None
    app_id = _resolve_app_id(pat, args.app_id, allow_prompt=not args.no_prompt)
    if not app_id:
        print("ERRO: App ID ausente para esta PAT.", file=sys.stderr)
        _print_app_id_help()
        return 1
    persist = args.save_binding or bool(args.app_id) or sys.stdin.isatty()
    session = DerivPatSession(
        pat,
        mode=args.mode,
        account_id=account,
        app_id=app_id,
    )
    try:
        result = await session.bootstrap(persist_binding=persist)
        ws_payload = await session.verify_websocket(result.ws_url)
    except DerivRestError as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        _print_app_id_help()
        return 1
    print("OK fluxo PAT completo")
    print(f"  app_id={result.app_id}")
    print(f"  conta={result.account_id} ({result.account_type}) saldo={result.balance}")
    print(f"  ws_url={result.ws_url}")
    print(f"  ws_time={ws_payload.get('time')}")
    for step in result.steps:
        print(f"  [{step.name}] {step.method} {step.path} -> {step.status}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Conecta Deriv via PAT (fluxo REST+OTP+WS)")
    parser.add_argument("--pat", default=None)
    parser.add_argument("--app-id", default=None, help="App ID do app PAT (ou use pat_...|APP_ID)")
    parser.add_argument("--account", default=None, help="Conta ex.: DOT92912876")
    parser.add_argument("--mode", default="demo", choices=("demo", "live"))
    parser.add_argument("--save-binding", action="store_true", help="Grava app_id em app/data/deriv/pat_bindings.json")
    parser.add_argument("--no-prompt", action="store_true", help="Nao pergunta App ID no terminal")
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
