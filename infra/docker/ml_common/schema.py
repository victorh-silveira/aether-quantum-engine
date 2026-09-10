from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Any


def canonical_schema_hash(names: Sequence[str]) -> str:
    blob = "\n".join(str(name) for name in names).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def validate_feature_vector(vector: Any, dim: int) -> list[float]:
    if not isinstance(vector, (list, tuple)):
        raise ValueError("feature_vector deve ser lista")
    if len(vector) != int(dim):
        raise ValueError(f"feature_vector deve ter {int(dim)} elementos, recebeu {len(vector)}")
    out: list[float] = []
    for value in vector:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("feature_vector contem valor nao finito")
        out.append(number)
    return out


def bundle_schema_hash_ok(bundle: dict[str, Any], expected: str) -> bool:
    stored = bundle.get("schema_hash")
    if stored is None or str(stored).strip() == "":
        return True
    return str(stored) == str(expected)


def request_schema_hash_ok(payload_hash: str | None, expected: str) -> bool:
    if payload_hash is None or str(payload_hash).strip() == "":
        return True
    return str(payload_hash) == str(expected)
