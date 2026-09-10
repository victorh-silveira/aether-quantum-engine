from __future__ import annotations

from collections import deque


class ContractIdDedupe:
    def __init__(self, *, max_ids: int = 4000) -> None:
        self._ids: set[str] = set()
        self._order: deque[str] = deque()
        self._max = max(1, int(max_ids))

    def seen_or_add(self, contract_id: str) -> bool:
        cid = str(contract_id or "").strip()
        if not cid:
            return False
        if cid in self._ids:
            return True
        self._ids.add(cid)
        self._order.append(cid)
        while len(self._order) > self._max:
            old = self._order.popleft()
            self._ids.discard(old)
        return False
