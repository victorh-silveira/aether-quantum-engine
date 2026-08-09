"""Cliente HTTP do loss-classifier (telemetria + veto fail-open)."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from src.domain.config_knobs import merge_settings_block, require_bool, require_float, require_int, require_keys
from src.infrastructure.inference.loss_classifier_types import (
    LossPredictRequest,
    LossPredictResponse,
    parse_loss_predict_response,
)


logger = logging.getLogger("AETH")


def resolve_loss_classifier_config(raw: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve infra.loss_classifier do SSOT."""
    block = merge_settings_block(("infra", "loss_classifier"), raw)
    require_keys(
        block,
        (
            "enabled",
            "http_url",
            "timeout_seconds",
            "max_connections",
            "max_keepalive_connections",
            "feature_dim",
            "veto_mode",
            "veto_p_loss_floor",
            "hard_p_loss_floor",
            "hard_blocks_pending_waive",
            "soft_kelly_mult",
            "soft_kelly_mult_high",
            "soft_p_loss_high",
            "soft_max_stake_pct_high",
            "ready_n",
            "retrain_min_n",
            "retrain_on_loss_min_n",
            "max_buffer",
            "flip_require_auto_learn",
            "flip_allow_seed_on_scale_discord",
            "flip_allow_seed_on_cal_discord",
        ),
        "infra.loss_classifier",
    )
    mode = str(block["veto_mode"]).strip().lower()
    if mode != "soft":
        raise ValueError("infra.loss_classifier.veto_mode deve ser soft (hard band via hard_p_loss_floor)")
    soft_mult = require_float(block, "soft_kelly_mult")
    if soft_mult <= 0.0 or soft_mult > 1.0:
        raise ValueError("infra.loss_classifier.soft_kelly_mult deve estar em (0, 1]")
    soft_high = require_float(block, "soft_kelly_mult_high")
    if soft_high <= 0.0 or soft_high > 1.0:
        raise ValueError("infra.loss_classifier.soft_kelly_mult_high deve estar em (0, 1]")
    if soft_high > soft_mult + 1e-12:
        raise ValueError("infra.loss_classifier.soft_kelly_mult_high deve ser <= soft_kelly_mult")
    p_high = require_float(block, "soft_p_loss_high")
    floor = require_float(block, "veto_p_loss_floor")
    hard_floor = require_float(block, "hard_p_loss_floor")
    if p_high <= floor:
        raise ValueError("infra.loss_classifier.soft_p_loss_high deve ser > veto_p_loss_floor")
    if hard_floor <= floor:
        raise ValueError("infra.loss_classifier.hard_p_loss_floor deve ser > veto_p_loss_floor")
    if hard_floor > 1.0:
        raise ValueError("infra.loss_classifier.hard_p_loss_floor deve estar em (veto_p_loss_floor, 1]")
    stake_pct = require_float(block, "soft_max_stake_pct_high")
    if stake_pct <= 0.0 or stake_pct > 0.05:
        raise ValueError("infra.loss_classifier.soft_max_stake_pct_high deve estar em (0, 0.05]")
    return {
        "enabled": require_bool(block, "enabled"),
        "http_url": str(block["http_url"]).rstrip("/"),
        "timeout_seconds": require_float(block, "timeout_seconds"),
        "max_connections": require_int(block, "max_connections"),
        "max_keepalive_connections": require_int(block, "max_keepalive_connections"),
        "feature_dim": require_int(block, "feature_dim"),
        "veto_mode": "soft",
        "veto_p_loss_floor": floor,
        "hard_p_loss_floor": hard_floor,
        "hard_blocks_pending_waive": require_bool(block, "hard_blocks_pending_waive"),
        "soft_kelly_mult": soft_mult,
        "soft_kelly_mult_high": soft_high,
        "soft_p_loss_high": p_high,
        "soft_max_stake_pct_high": stake_pct,
        "ready_n": require_int(block, "ready_n"),
        "retrain_min_n": require_int(block, "retrain_min_n"),
        "retrain_on_loss_min_n": require_int(block, "retrain_on_loss_min_n"),
        "max_buffer": require_int(block, "max_buffer"),
        "flip_require_auto_learn": require_bool(block, "flip_require_auto_learn"),
        "flip_allow_seed_on_scale_discord": require_bool(block, "flip_allow_seed_on_scale_discord"),
        "flip_allow_seed_on_cal_discord": require_bool(block, "flip_allow_seed_on_cal_discord"),
    }


def loss_classifier_enabled(config: dict[str, Any] | None) -> bool:
    """Indica se loss-classifier esta habilitado na config raiz."""
    if not isinstance(config, dict):
        return bool(resolve_loss_classifier_config(None)["enabled"])
    infra = config.get("infra")
    chunk = infra.get("loss_classifier") if isinstance(infra, dict) else None
    if isinstance(chunk, dict) and "enabled" in chunk:
        return bool(chunk["enabled"])
    return bool(resolve_loss_classifier_config(None)["enabled"])


class LossClassifierClient:
    """Cliente httpx assincrono para predict/learn do loss-classifier."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout: float,
        enabled: bool,
        veto_p_loss_floor: float,
        max_connections: int = 8,
        max_keepalive_connections: int = 4,
    ) -> None:
        self._enabled = bool(enabled)
        self._veto_floor = float(veto_p_loss_floor)
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(float(timeout)),
            limits=httpx.Limits(
                max_connections=int(max_connections),
                max_keepalive_connections=int(max_keepalive_connections),
            ),
        )

    @property
    def enabled(self) -> bool:
        """True se chamadas remotas estao ativas."""
        return self._enabled

    async def aclose(self) -> None:
        """Fecha o cliente HTTP."""
        await self._client.aclose()

    async def predict_loss(self, request: LossPredictRequest) -> LossPredictResponse:
        """POST /v1/predict_loss; fail-open sem veto."""
        empty: LossPredictResponse = {
            "p_loss": 0.5,
            "veto": False,
            "auto_learn_applied": False,
            "model_version": "none",
            "n_train": 0,
            "veto_ready": False,
            "bootstrap": False,
        }
        if not self._enabled:
            return empty
        payload = {
            "feature_vector": [float(v) for v in request["feature_vector"]],
            "symbol": str(request.get("symbol") or ""),
            "direction": str(request.get("direction") or ""),
            "veto_p_loss_floor": float(request.get("veto_p_loss_floor") or self._veto_floor),
        }
        try:
            response = await self._client.post("/v1/predict_loss", json=payload)
            response.raise_for_status()
            return parse_loss_predict_response(response.json())
        except (httpx.TimeoutException, httpx.HTTPError, ValueError, TypeError, KeyError) as exc:
            logger.warning("LOSS_CLF || FALLBACK predict | %s", exc)
            return empty

    async def learn(
        self,
        *,
        feature_vector: list[float],
        label: str,
        contract_id: str = "",
        symbol: str = "",
    ) -> dict[str, Any]:
        """POST /v1/learn fail-open."""
        if not self._enabled:
            return {"ok": False, "skipped": True}
        payload = {
            "feature_vector": [float(v) for v in feature_vector],
            "label": str(label).upper(),
            "contract_id": str(contract_id),
            "symbol": str(symbol),
        }
        try:
            response = await self._client.post("/v1/learn", json=payload)
            response.raise_for_status()
            body = response.json()
            return body if isinstance(body, dict) else {"ok": True}
        except (httpx.TimeoutException, httpx.HTTPError, ValueError, TypeError) as exc:
            logger.warning("LOSS_CLF || FALLBACK learn | %s", exc)
            return {"ok": False, "error": str(exc)}


def build_loss_classifier_client_from_config(config: dict[str, Any] | None) -> LossClassifierClient:
    """Factory a partir da config raiz do motor."""
    cfg = resolve_loss_classifier_config(None)
    if isinstance(config, dict):
        infra = config.get("infra")
        raw = infra.get("loss_classifier") if isinstance(infra, dict) else None
        if isinstance(raw, dict):
            cfg = resolve_loss_classifier_config(raw)
    url = str(cfg["http_url"] or os.getenv("AETHER_LOSS_CLASSIFIER_HTTP", "http://localhost:8006"))
    return LossClassifierClient(
        base_url=url,
        timeout=float(cfg["timeout_seconds"]),
        enabled=bool(cfg["enabled"]),
        veto_p_loss_floor=float(cfg["veto_p_loss_floor"]),
        max_connections=int(cfg["max_connections"]),
        max_keepalive_connections=int(cfg["max_keepalive_connections"]),
    )
