"""Fluxo PAT: health, accounts, OTP e verificacao WebSocket."""

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from aether_paths import REPO_ROOT
from src.infrastructure.api.deriv_http import read_http_response
from src.infrastructure.api.deriv_pat_binding import (
    DerivPatBindingError,
    discover_app_id_for_pat,
    parse_deriv_pat,
    save_binding,
)
from src.infrastructure.api.deriv_rest_client import (
    DerivRestClient,
    DerivRestError,
    select_account,
)
from src.infrastructure.api.websocket_manager import WebSocketManager


_TRANSIENT_HTTP = frozenset({502, 503, 504})


@dataclass(frozen=True)
class PatFlowStep:
    """Registro de uma etapa do bootstrap PAT para diagnostico."""

    name: str
    method: str
    path: str
    status: int
    response_preview: str


@dataclass(frozen=True)
class PatBootstrapResult:
    """Resultado consolidado apos health, accounts e OTP."""

    pat: str
    app_id: str
    account_id: str
    account_type: str
    balance: float
    ws_url: str
    steps: tuple[PatFlowStep, ...]


class DerivPatSession:
    """Orquestra health REST, resolucao de App ID e emissao de OTP WS."""

    def __init__(
        self,
        pat: str,
        *,
        mode: str = "demo",
        account_id: str | None = None,
        rest_base_url: str = "https://api.derivws.com",
        app_id: str | None = None,
        timeout_seconds: int = 60,
        logger: logging.Logger | None = None,
    ):
        token, _ = parse_deriv_pat(pat)
        self.pat = token
        self.mode = mode
        self.account_id_override = account_id
        self.rest_base_url = rest_base_url
        self.app_id_explicit = app_id
        self.timeout_seconds = timeout_seconds
        self.logger = logger or logging.getLogger("AETH")
        self._steps: list[PatFlowStep] = []

    def _log_step(self, name: str, method: str, path: str, status: int, body: str) -> None:
        """Armazena e registra um passo do fluxo PAT."""
        preview = body.replace("\n", " ")[:500]
        self._steps.append(PatFlowStep(name, method, path, status, preview))
        self.logger.info("PAT %s %s %s -> %s %s", name, method, path, status, preview)

    def resolve_app_id(self) -> str:
        """Resolve App ID via binding, config ou candidatos."""
        return discover_app_id_for_pat(
            self.pat,
            REPO_ROOT,
            rest_base=self.rest_base_url,
            timeout=float(self.timeout_seconds),
            explicit=self.app_id_explicit,
        )

    def _client(self, app_id: str) -> DerivRestClient:
        """Instancia cliente REST autenticado com PAT e App ID."""
        return DerivRestClient(
            rest_base_url=self.rest_base_url,
            deriv_app_id=app_id,
            access_token=self.pat,
            timeout_seconds=self.timeout_seconds,
        )

    def health_check(self, *, retries: int = 4, retry_delay: float = 1.5) -> str:
        """Chama GET /v1/health na API Deriv com retry em 502/503/504."""
        path = "/v1/health"
        url = f"{self.rest_base_url.rstrip('/')}{path}"
        req = urllib.request.Request(url, method="GET")
        attempts = max(1, int(retries))
        last_exc: BaseException = RuntimeError("PAT health_check sem resposta")
        for attempt in range(attempts):
            try:
                body = read_http_response(req, float(self.timeout_seconds)).decode("utf-8")
                self._log_step("health", "GET", path, 200, body)
                return body
            except urllib.error.HTTPError as exc:
                last_exc = exc
                self._log_step("health", "GET", path, int(exc.code), str(exc.reason))
                if int(exc.code) not in _TRANSIENT_HTTP:
                    raise
            except urllib.error.URLError as exc:
                last_exc = exc
                self._log_step("health", "GET", path, 0, str(exc.reason))
            if attempt + 1 >= attempts:
                break
            time.sleep(float(retry_delay) * float(attempt + 1))
        raise last_exc

    async def bootstrap(
        self,
        *,
        persist_binding: bool = False,
    ) -> PatBootstrapResult:
        """Executa health, accounts, OTP e opcionalmente persiste binding."""
        self._steps.clear()
        try:
            self.health_check()
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
            code = int(getattr(exc, "code", 0) or 0)
            self.logger.warning(
                "PAT health indisponivel (HTTP %s: %s); seguindo com accounts/OTP.",
                code or "n/a",
                exc,
            )
        try:
            app_id = self.resolve_app_id()
        except DerivPatBindingError as exc:
            raise DerivRestError(str(exc)) from exc
        client = self._client(app_id)
        try:
            accounts = await client.list_accounts()
        except DerivRestError as exc:
            self._log_step("accounts", "GET", "/trading/v1/options/accounts", 0, str(exc))
            raise
        self._log_step(
            "accounts",
            "GET",
            "/trading/v1/options/accounts",
            200,
            json.dumps([a.account_id for a in accounts]),
        )
        account = select_account(accounts, self.mode, self.account_id_override)
        ws_url = await client.request_otp_ws_url(account.account_id)
        self._log_step(
            "otp",
            "POST",
            f"/trading/v1/options/accounts/{account.account_id}/otp",
            200,
            json.dumps({"url": ws_url}),
        )
        if persist_binding:
            save_binding(REPO_ROOT, self.pat, app_id)
        return PatBootstrapResult(
            pat=self.pat,
            app_id=app_id,
            account_id=account.account_id,
            account_type=account.account_type,
            balance=account.balance,
            ws_url=ws_url,
            steps=tuple(self._steps),
        )

    async def verify_websocket(self, ws_url: str) -> dict[str, Any]:
        """Conecta no WS OTP e envia ping time."""
        ws = WebSocketManager(ws_url, request_timeout=self.timeout_seconds)
        await ws.connect()
        try:
            payload = await ws.send({"time": 1, "req_id": 99})
            self._log_step("ws_time", "WS", "time", 200, json.dumps(payload)[:500])
            return payload
        finally:
            await ws.close()
