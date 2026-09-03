---
name: aether-architecture-senior
description: >-
  Audita e orienta mudancas na arquitetura senior Aether (host Python 3.13,
  DDD/hexagonal, asyncio, Polars SSOT, sidecars ML, Docker core/ml). Use when
  designing layers, event-loop/CUDA offload, ports/adapters, or reviewing
  architecture PRs.
---

# Arquitetura sênior

## Quando aplicar

Mudanca de camadas DDD, ports/adapters, event loop, CUDA/host, Polars, sidecars HTTP, compose `core,ml`, ou review de PR de arquitetura.

## Checklist

1. Ler `docs/engineering-architecture-senior.md` + `docs/arquitetura.md` (pipeline runtime)
2. Confirmar dominio puro (sem I/O / sem import de infra)
3. Application so via ports; adapters em `infrastructure/`
4. Hot path asyncio: offload de PyTorch/Polars pesado; sem bloqueio do loop WS
5. TCN no host; meta `:8005` / loss `:8006` com timeout e fallback conforme settings
6. Polars-only; sem pandas
7. Settlement: preservar ZSET `settlement:queue:priority` salvo mandato explicito
8. Segredos fora do git; Docker em `127.0.0.1`
9. Fechar com skill `aether-surface-sync` se a superficie mudou

## Nunca

- Mover I/O para `domain/`
- Bloquear o event loop no hot path
- Expor portas fora de loopback
- Introduzir pandas ou afrouxar QA 100% / 300 linhas

## Skills irmas

`aether-infra-stack`, `aether-dl-train`, `aether-deriv-connect`, `aether-python-deps`, `aether-precommit`, `aether-surface-sync`
