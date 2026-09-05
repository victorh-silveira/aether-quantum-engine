"""Allowlist Vulture — Protocols hexagonais e simbolos registrados por reflexao.

Regenerar (append) apos auditar falsos positivos:
  cd app && python -m vulture src run.py train.py scripts aether_paths.py --min-confidence 80 --make-whitelist

Itens que nao forem porta Protocol / hook de registro devem ser removidos do codigo, nao preservados aqui.
"""

MarketCandlePort.get_latest_candle  # unused method (typing.Protocol)
MarketCandlePort.stream_candles  # unused method (typing.Protocol)
SettlementQueuePort.enqueue  # unused method (typing.Protocol)
SettlementQueuePort.pop_due  # unused method (typing.Protocol)
ModelArtifactPort.load_bytes  # unused method (typing.Protocol)
ModelArtifactPort.put_bytes  # unused method (typing.Protocol)
