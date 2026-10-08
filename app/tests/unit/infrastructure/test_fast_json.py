"""Testes unitarios para o modulo de serializacao fast_json."""

import importlib
import sys
from types import ModuleType
from unittest.mock import MagicMock

from src.infrastructure.serialization import fast_json


def test_fast_json_roundtrip_primitives():
    data = {"int": 42, "float": 3.14159, "str": "1HZ75V", "bool": True, "none": None}
    serialized = fast_json.dumps(data)
    assert isinstance(serialized, str)
    deserialized = fast_json.loads(serialized)
    assert deserialized == data


def test_fast_json_dump_bytes_and_loads():
    data = {"symbol": "1HZ75V", "cycle": 10, "metrics": [1.0, 2.0, 3.5]}
    raw_bytes = fast_json.dump_bytes(data)
    assert isinstance(raw_bytes, bytes)
    deserialized = fast_json.loads(raw_bytes)
    assert deserialized == data


def test_fast_json_is_orjson_available_returns_bool():
    available = fast_json.is_orjson_available()
    assert isinstance(available, bool)


def test_fast_json_mock_orjson_branch(monkeypatch):
    mock_orjson = MagicMock()
    mock_orjson.dumps.return_value = b'{"mock":1}'
    mock_orjson.loads.return_value = {"mock": 1}
    mock_orjson.OPT_NON_STR_KEYS = 1

    monkeypatch.setattr(fast_json, "_HAS_ORJSON", True)
    monkeypatch.setattr(fast_json, "orjson", mock_orjson)

    res_str = fast_json.dumps({"test": 123})
    assert res_str == '{"mock":1}'

    res_bytes = fast_json.dump_bytes({"test": 123})
    assert res_bytes == b'{"mock":1}'

    res_obj = fast_json.loads(b'{"mock":1}')
    assert res_obj == {"mock": 1}


def test_fast_json_reload_with_orjson(monkeypatch):
    mock_mod = ModuleType("orjson")
    mock_mod.dumps = MagicMock(return_value=b'{"ok":true}')
    mock_mod.loads = MagicMock(return_value={"ok": True})
    mock_mod.OPT_NON_STR_KEYS = 1

    monkeypatch.setitem(sys.modules, "orjson", mock_mod)
    importlib.reload(fast_json)
    assert fast_json.is_orjson_available() is True
    assert fast_json.dumps({"ok": True}) == '{"ok":true}'

    monkeypatch.delitem(sys.modules, "orjson", raising=False)
    importlib.reload(fast_json)
