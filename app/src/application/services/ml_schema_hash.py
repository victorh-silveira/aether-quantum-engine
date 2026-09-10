"""Hash canonico de schema tabular compartilhado com sidecars ML."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence


def canonical_schema_hash(names: Sequence[str]) -> str:
    """SHA-256 dos nomes na ordem canonica (mesmo algoritmo dos sidecars)."""
    blob = "\n".join(str(name) for name in names).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
