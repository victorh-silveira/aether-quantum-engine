"""Serializacao ultrarrapida com fallback transparente entre orjson e json."""

from __future__ import annotations

import importlib
import json
from typing import Any


try:
    orjson = importlib.import_module("orjson")
    _HAS_ORJSON = True
except ImportError:
    orjson = None
    _HAS_ORJSON = False


def dumps(obj: Any) -> str:
    """Serializa objeto para string JSON usando orjson ou fallback padrao."""
    if _HAS_ORJSON and orjson is not None:
        return str(orjson.dumps(obj, option=orjson.OPT_NON_STR_KEYS).decode("utf-8"))
    return json.dumps(obj, separators=(",", ":"))


def dump_bytes(obj: Any) -> bytes:
    """Serializa objeto para bytes JSON usando orjson ou fallback padrao."""
    if _HAS_ORJSON and orjson is not None:
        return bytes(orjson.dumps(obj, option=orjson.OPT_NON_STR_KEYS))
    return json.dumps(obj, separators=(",", ":")).encode("utf-8")


def loads(data: str | bytes) -> Any:
    """Desserializa JSON usando orjson ou fallback padrao."""
    if _HAS_ORJSON and orjson is not None:
        return orjson.loads(data)
    return json.loads(data)


def is_orjson_available() -> bool:
    """Indica se a biblioteca orjson esta disponivel no runtime."""
    return _HAS_ORJSON


__all__ = ["dump_bytes", "dumps", "is_orjson_available", "loads"]
