"""Cliente REST Deriv (PAT Bearer + OTP para WebSocket)."""

import asyncio
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from src.infrastructure.api.deriv_http import read_http_response


class DerivRestError(Exception):
    """Falha em chamada REST Deriv."""


@dataclass(frozen=True)
class DerivAccount:
    """Conta Options retornada por GET /trading/v1/options/accounts."""

    account_id: str
    balance: float
    account_type: str
    status: str
    currency: str


@dataclass(frozen=True)
class DerivTradingSession:
    """Sessao pronta para conectar WebSocket autenticado."""

    ws_url: str
    balance: float
    account_id: str


def _account_type_for_mode(mode: str) -> str:
    """Mapeia modo do motor (demo/live) para account_type da Deriv."""
    m = mode.lower()
    if m in ("live", "real"):
        return "real"
    return "demo"


def select_account(
    accounts: list[DerivAccount],
    mode: str,
    account_id_override: str | None = None,
) -> DerivAccount:
    """Escolhe conta ativa por modo ou ID explicito."""
    if account_id_override:
        for acc in accounts:
            if acc.account_id == account_id_override:
                return acc
        raise DerivRestError(f"Conta nao encontrada: {account_id_override}")
    want = _account_type_for_mode(mode)
    active = [a for a in accounts if a.status == "active" and a.account_type == want]
    if not active:
        active = [a for a in accounts if a.account_type == want]
    if not active:
        tipos = sorted({a.account_type for a in accounts})
        raise DerivRestError(f"Nenhuma conta Deriv tipo={want} (disponiveis: {tipos})")
    return active[0]


class DerivRestClient:
    """HTTP REST para contas e OTP WebSocket."""

    def __init__(
        self,
        *,
        rest_base_url: str,
        deriv_app_id: str,
        access_token: str,
        timeout_seconds: int = 60,
    ):
        self.rest_base_url = rest_base_url.rstrip("/")
        self.deriv_app_id = deriv_app_id
        self.access_token = access_token
        self.timeout_seconds = timeout_seconds

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Executa chamada REST autenticada e retorna JSON como dict."""
        url = f"{self.rest_base_url}{path}"
        data = None
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Deriv-App-ID": self.deriv_app_id,
            "Accept": "application/json",
        }
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            raw = read_http_response(req, float(self.timeout_seconds)).decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise DerivRestError(f"{method} {path} HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise DerivRestError(f"{method} {path} falhou: {exc}") from exc
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise DerivRestError(f"Resposta JSON invalida em {path}")
        return parsed

    async def list_accounts(self) -> list[DerivAccount]:
        """Lista contas Options (GET /trading/v1/options/accounts)."""
        payload = await asyncio.to_thread(
            self._request,
            "GET",
            "/trading/v1/options/accounts",
        )
        rows = payload.get("data")
        if not isinstance(rows, list):
            raise DerivRestError("Campo data ausente em list_accounts")
        out: list[DerivAccount] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            aid = row.get("account_id")
            if not aid:
                continue
            out.append(
                DerivAccount(
                    account_id=str(aid),
                    balance=float(row.get("balance") or 0),
                    account_type=str(row.get("account_type") or ""),
                    status=str(row.get("status") or ""),
                    currency=str(row.get("currency") or "USD"),
                )
            )
        if not out:
            raise DerivRestError("Nenhuma conta retornada pela Deriv")
        return out

    async def request_otp_ws_url(self, account_id: str) -> str:
        """Obtem URL WebSocket com OTP (POST .../accounts/{id}/otp)."""
        path = f"/trading/v1/options/accounts/{urllib.parse.quote(account_id, safe='')}/otp"
        payload = await asyncio.to_thread(self._request, "POST", path)
        data = payload.get("data")
        if not isinstance(data, dict):
            raise DerivRestError("Campo data ausente em otp")
        url = data.get("url")
        if not url or not isinstance(url, str):
            raise DerivRestError("URL WebSocket OTP ausente na resposta")
        return url

    async def post_otp(self, account_id: str) -> str:
        """Renova OTP de uso unico e retorna URL WebSocket autorizada."""
        return await self.request_otp_ws_url(account_id)

    async def open_trading_session(
        self,
        mode: str,
        account_id_override: str | None = None,
    ) -> DerivTradingSession:
        """Resolve conta, saldo e URL WS autenticada."""
        accounts = await self.list_accounts()
        account = select_account(accounts, mode, account_id_override)
        ws_url = await self.request_otp_ws_url(account.account_id)
        return DerivTradingSession(
            ws_url=ws_url,
            balance=account.balance,
            account_id=account.account_id,
        )
