---
name: aether-deriv-connect
description: >-
  Diagnostica conexao Deriv no Aether (PAT+OTP, WSS handshake, Cloudflare/5xx,
  stream dual-timeframe). Use when AUTH fails, WSS handshake errors, stream
  reconnect loops, or the user mentions PAT, OTP, or api.derivws.com.
---

# Deriv connect

## Passos

1. Credenciais PAT — nao logar segredo; ver `docs/deriv-api-aether.md`
2. Handshake WSS host/IP; retries Cloudflare/5xx
3. AUTH saldo/conta; trading via WSS
4. Stream micro+macro; tick buffer
5. Reconnect: backoff settings; nao reduzir open_timeout sem evidencia

Nunca commitar tokens. Referencia longa: `docs/deriv-api.md`
