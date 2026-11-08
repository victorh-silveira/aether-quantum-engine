"""Pacote de serializacao de alta velocidade."""

from .fast_json import dump_bytes, dumps, is_orjson_available, loads


__all__ = ["dump_bytes", "dumps", "is_orjson_available", "loads"]
