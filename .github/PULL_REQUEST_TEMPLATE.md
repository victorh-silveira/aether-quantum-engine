## Summary

<!-- O que muda e por que (1-3 bullets). -->

## Crivo de higiene

| Vetor | OK? | Notas |
|-------|-----|--------|
| Dead code | | vulture 80 + allowlist Protocol-only; sem ERA/ARG/T20 novos |
| Feature flags | | temporarias em `TEMPORARY_FLAGS` com expiry; sem bifurcar legado |
| Async leaks | | sem `create_task` orfao; TaskGroup/supervisor |
| Redis TTL | | TTL so efemeros (`bar_sig`/`market_sig`); settlement ZSET **sem** TTL/Streams |
| DB | | sem indice/chunk/policy sem mandato |
| I/O sync | | sem `requests`/`pandas`/`print` no motor |

## Test plan

- [ ] Pre-commit WSL verde (`make app-pre-commit-run`)
- [ ] Testes novos/alterados cobrem o contrato mudado
- [ ] Surface-sync se docs/rules/skills/AGENTS tocados
