---
name: aether-deriv-connect
description: >-
  Diagnostica conexao Deriv no Aether (PAT+OTP, WSS handshake, Cloudflare/5xx,
  stream dual-timeframe). Use when AUTH fails, WSS handshake errors, stream
  reconnect loops, or the user mentions PAT, OTP, or api.derivws.com.
---

# Deriv connect

## Passos

1. Credenciais PAT — nao logar segredo; ver `docs/deriv-api-aether.md` (env/secret store)
2. Handshake WSS host/IP; retries Cloudflare/5xx; heartbeat + reconexao com backoff/jitter
3. AUTH saldo/conta; trading via WSS; REST via httpx idempotente
4. Stream micro+macro; tick buffer; backpressure: preferir latest price
5. Reconnect: backoff settings; nao reduzir open_timeout sem evidencia

Nunca commitar tokens. Referencia longa: `docs/deriv-api.md`. Arquitetura: `docs/engineering-architecture-senior.md`
