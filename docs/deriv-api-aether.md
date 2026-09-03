# Deriv API — guia rapido para agentes (Aether)

Referencia completa: [deriv-api.md](deriv-api.md). Arquitetura (WS/httpx/backpressure): [engineering-architecture-senior.md](engineering-architecture-senior.md).

Use ONLY this API. Symbol field: `underlying_symbol`.

## Surfaces

| Surface | Gateway | Auth |
| --- | --- | --- |
| Public market data | `wss://api.derivws.com/trading/v1/options/ws/public` | None |
| Authenticated trading WS | OTP URL from `POST /trading/v1/options/accounts/{accountId}/otp` | OTP in URL only |
| Bulk-purchase REST | `POST /trading/v1/options/contracts/bulk-purchase/{demo,real}` | `Deriv-App-ID` + PAT in body (never Bearer) |
| REST accounts / OTP | `https://api.derivws.com` | `Deriv-App-ID` + `Authorization: Bearer <token>` |

## Aether mapping

- Treino DL / meta / historico: sempre WSS publico.
- Execucao: tenta WSS OTP; se handshake falhar, WSS publico + bulk-purchase REST.
- Liquidacao REST: apos expiry, saldo via accounts + payload sintetico (`settlement_rest`).
- Credenciais: `AETHER_DERIV_PAT`, `AETHER_DERIV_APP_ID`.

## Key rules

1. Public market data: `/ws/public` sem OTP.
2. Trading WSS: emitir OTP via REST, conectar na URL; sem auth no socket.
3. Bulk-purchase: App-ID no header + PAT no body — nunca Bearer.
4. OTP one-shot: renovar antes de cada reconnect / failover de IP.
