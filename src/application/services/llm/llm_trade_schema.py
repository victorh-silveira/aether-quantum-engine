"""Schema JSON Gemini para decisao CALL/PUT obrigatoria."""

from __future__ import annotations

from typing import Any


def apply_trade_json_output(cfg_base: dict[str, Any], types_module: Any) -> None:
    """Forca response_mime_type application/json com campos CALL/PUT."""
    call_put = types_module.Schema(type=types_module.Type.STRING, enum=["CALL", "PUT"])
    cfg_base["response_mime_type"] = "application/json"
    cfg_base["response_schema"] = types_module.Schema(
        type=types_module.Type.OBJECT,
        properties={
            "EURUSD": call_put,
            "US_CLUSTER": call_put,
            "EU_CLUSTER": call_put,
            "Probabilidade": types_module.Schema(type=types_module.Type.NUMBER),
        },
        required=["EURUSD", "US_CLUSTER", "EU_CLUSTER", "Probabilidade"],
    )
