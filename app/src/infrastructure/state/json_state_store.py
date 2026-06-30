"""StateStore em arquivo JSON para testes e modo sem infra."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aether_paths import repo_path
from src.domain.risk.recovery_hurst_decay import REDIS_SKIP_COUNTER_KEY


class JsonStateStore:
    """Persistencia local compativel com a porta StateStore."""

    def __init__(self, file_path: str | Path | None = None):
        self._path = Path(file_path) if file_path is not None else repo_path("data", "state.json")
        self._strings: dict[str, str] = {}
        self._hashes: dict[str, dict[str, str]] = {}
        self._path.parent.mkdir(parents=True, exist_ok=True)

    async def save_snapshot(self, payload: dict[str, Any]) -> None:
        """Grava snapshot completo em arquivo JSON."""
        with self._path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    async def save_state_bundle(
        self,
        *,
        snapshot: dict[str, Any],
        session: dict[str, Any] | None = None,
        market_sig: str | None = None,
        recovery_skip_counter: int | None = None,
    ) -> None:
        """Compativel com RedisStateStore; persiste snapshot e hashes em memoria."""
        await self.save_snapshot(snapshot)
        risk = snapshot.get("risk")
        if isinstance(risk, dict):
            flat = {str(k): str(v) for k, v in risk.items() if not isinstance(v, (dict, list))}
            if flat:
                await self.set_hash("state:risk", flat)
            pending = risk.get("pending_loss")
            if isinstance(pending, dict):
                await self.set_hash("state:pending_loss", pending)
        if session:
            await self.set_hash("session:daily", session)
        if market_sig:
            await self.set_string("market_sig", market_sig)
        if recovery_skip_counter is not None:
            await self.set_string(REDIS_SKIP_COUNTER_KEY, str(max(0, int(recovery_skip_counter))))

    async def load_snapshot(self) -> dict[str, Any] | None:
        """Carrega snapshot do arquivo JSON."""
        if not self._path.exists() or self._path.stat().st_size == 0:
            return None
        try:
            with self._path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else None
        except (json.JSONDecodeError, OSError):
            return None

    async def set_hash(self, key: str, mapping: dict[str, Any]) -> None:
        """Armazena hash em memoria para testes."""
        self._hashes[key] = {str(k): str(v) for k, v in mapping.items()}

    async def get_hash(self, key: str) -> dict[str, str]:
        """Retorna hash armazenado em memoria."""
        return dict(self._hashes.get(key, {}))

    async def set_string(self, key: str, value: str, *, ttl_seconds: int | None = None) -> None:
        """Armazena string em memoria ignorando TTL."""
        _ = ttl_seconds
        self._strings[key] = str(value)

    async def get_string(self, key: str) -> str | None:
        """Retorna string armazenada em memoria."""
        return self._strings.get(key)

    async def ping(self) -> bool:
        """Sempre disponivel em modo local."""
        return True

    async def close(self) -> None:
        """Encerramento sem efeito."""
        return

    def save(self, data: dict[str, Any]) -> None:
        """Compatibilidade sync com PersistenceManager em testes."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)

    def load(self) -> dict[str, Any] | None:
        """Compatibilidade sync com PersistenceManager em testes."""
        if not self._path.exists() or self._path.stat().st_size == 0:
            return None
        try:
            with self._path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (json.JSONDecodeError, OSError):
            return None
