"""Reabastece saldo da conta demo Deriv via reset-demo-balance."""

import argparse
import asyncio
import json
import sys
from pathlib import Path


_APP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_APP))

from dotenv import load_dotenv

from aether_paths import REPO_ROOT
from src.application.services.auth_manager import AuthManager
from src.infrastructure.api.deriv_rest_client import DerivRestError, select_account


async def _run(config_path: str | None) -> int:
    """Carrega config, reseta saldo demo e imprime resultado."""
    load_dotenv(REPO_ROOT / ".env")
    cfg: dict = {}
    if config_path:
        with Path(config_path).open(encoding="utf-8") as fh:
            cfg = json.load(fh)
    auth = AuthManager(mode="demo", config=cfg)
    client = auth.rest_client()
    accounts = await client.list_accounts()
    account = select_account(accounts, "demo", auth.account_id_override)
    path = f"/trading/v1/options/accounts/{account.account_id}/reset-demo-balance"
    payload = await asyncio.to_thread(client._request, "POST", path)
    data = payload.get("data") or {}
    balance = float(data.get("balance") or 0.0)
    print(json.dumps(data, indent=2))
    print(f"Conta demo {account.account_id} reabastecida: saldo=${balance:.2f} {data.get('currency', 'USD')}")
    return 0


def main() -> int:
    """Ponto de entrada CLI para reset de saldo demo."""
    parser = argparse.ArgumentParser(description="Reabastece saldo da conta demo Deriv (reset para ~$10000)")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "config" / "settings.json"),
        help="Caminho do settings.json (api_config.rest_base_url)",
    )
    args = parser.parse_args()
    try:
        return asyncio.run(_run(args.config))
    except DerivRestError as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
