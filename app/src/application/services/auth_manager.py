"""Gerenciamento de autenticacao PAT e sessao Deriv (REST + OTP)."""

import logging
import os
from typing import Any

from dotenv import load_dotenv

from aether_paths import APP_ROOT, REPO_ROOT
from src.infrastructure.api.deriv_credentials import is_legacy_deriv_app_id, resolve_deriv_app_id
from src.infrastructure.api.deriv_pat_binding import parse_deriv_pat
from src.infrastructure.api.deriv_rest_client import DerivRestClient, DerivRestError, DerivTradingSession


class AuthManager:
    """PAT Bearer, REST accounts/OTP e WebSocket autenticado."""

    def __init__(self, mode: str = "demo", config: dict[str, Any] | None = None):
        for env_path in (REPO_ROOT / ".env", APP_ROOT / ".env"):
            if env_path.is_file():
                load_dotenv(env_path)
        self.mode = mode
        self.config = config or {}
        api = self.config.get("api_config") or {}
        self.rest_base_url = str(api.get("rest_base_url") or "https://api.derivws.com")
        self.deriv_app_id = ""
        self.account_id_override = os.getenv("AETHER_DERIV_ACCOUNT_ID") or os.getenv("AETHER_OAUTH_ACCOUNT_ID") or None
        self.request_timeout = int(api.get("request_timeout_seconds") or 60)
        self.logger = logging.getLogger("AETH")
        self.logger.debug("AUTH | modo=%s PAT", mode.upper())

    def get_pat(self) -> str | None:
        """Retorna o token PAT limpo a partir do ambiente."""
        raw = os.getenv("AETHER_DERIV_PAT")
        if not raw or not raw.strip():
            return None
        token, _ = parse_deriv_pat(raw.strip())
        return token or None

    def _ensure_deriv_app_id(self) -> str:
        """Resolve e cacheia o App ID Deriv para esta instancia."""
        if self.deriv_app_id:
            return self.deriv_app_id
        pat = self.get_pat()
        self.deriv_app_id = resolve_deriv_app_id(
            config=self.config,
            repo_root=REPO_ROOT,
            pat=pat,
        )
        return self.deriv_app_id

    def rest_client(self) -> DerivRestClient:
        """Monta cliente REST autenticado com PAT e App ID validos."""
        token = self.get_pat()
        if not token:
            raise DerivRestError(
                "AETHER_DERIV_PAT ausente. Valide com: python app/scripts/operations/deriv_pat_connect.py"
            )
        app_id = self._ensure_deriv_app_id()
        if not app_id:
            raise DerivRestError("AETHER_DERIV_APP_ID ausente (config/deriv_pat_app_id ou pat_...|APP_ID no .env)")
        if is_legacy_deriv_app_id(app_id):
            raise DerivRestError(
                f"AETHER_DERIV_APP_ID={app_id} e legado; use App ID do app PAT em developers.deriv.com"
            )
        return DerivRestClient(
            rest_base_url=self.rest_base_url,
            deriv_app_id=app_id,
            access_token=token,
            timeout_seconds=self.request_timeout,
        )

    async def open_trading_session(self) -> DerivTradingSession:
        """Abre sessao de trading (accounts + OTP) para o modo configurado."""
        client = self.rest_client()
        return await client.open_trading_session(self.mode, self.account_id_override)
