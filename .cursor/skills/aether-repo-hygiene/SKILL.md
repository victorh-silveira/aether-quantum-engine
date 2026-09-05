---
name: aether-repo-hygiene
description: >-
  Planeja e executa higienizacao do repositorio Aether (codigo/scripts/containers/
  docs/skills/JSON/git/Makefile/deps mortos, vulture allowlist, Redis TTL). Use when
  the user asks for purge, dead code, orphan docs/skills, unused containers, JSON
  SSOT cleanup, or full repo hygiene.
---

# Higienizacao do repositorio

## Antes de apagar

1. Ler `docs/engineering-repo-hygiene.md` + `docs/agent-coverage.md` + `AGENTS.md`
2. Inventariar candidatos; classificar **morto comprovado** vs vivo frio vs indexado
3. Rodar vulture com `app/.vulture_whitelist.py` (`--min-confidence 80`); ruff F/ERA/ARG/T20
4. Montar plan em ondas (codigo → JSON → alinhamentos → deps); fora de escopo explicito

## Checklist do guia (tooling + sidecars)

- [ ] Vulture CI: confidence **80** + whitelist no `clean_workspace` / `pyproject`
- [ ] Allowlist so Protocols/hooks (sem negocio)
- [ ] Redis TTL so em efemeros (`bar_sig` / `market_sig` / `corr_matrix`); **sem** TTL em ZSET settlement / `session:current*`
- [ ] Anti-Streams no settlement
- [ ] Feature flags temporarias em `config_flag_ttl.TEMPORARY_FLAGS` com expiry; sentinela verde
- [ ] Async: sem `create_task` orfao; preferir TaskGroup / supervisor
- [ ] MinIO ILM `optuna/` ~7d; bucket `dl-models`
- [ ] WSL: `clean_env.sh` + prune/fstrim; compactar VHDX manual trimestral (sem diskpart no CI)

## Execucao

1. Onda 1: modulos/scripts sem callers; lixo `.bak`; gitignore/workflows/Makefile; gap `AGENTS` so se skill viva
2. Onda 2: chaves JSON sem leitor/`resolve_*`; doctrine SSOT; docs settings
3. Onda 3: hydrate/TF legado; CHANGELOG Unreleased; nomenclatura vs SSOT
4. Deps: skill `aether-python-deps` (nao purge cego de pip)
5. Cursor/VS Code: so orfaos fora da matriz; manter doutrina/engenharia `alwaysApply`
6. Apos cada onda: `make app-pre-commit-run` (WSL); commits separados PT-BR

## Evidencia por delete

Grep de path/simbolo; ausencia em compose/Makefile/matriz; vulture limpo; testes doctrine/policy se cabivel.

## Nunca

Apagar linha da matriz `agent-coverage`; remover redis/timescale/minio/meta/loss; refatorar hexagonal por estetica; revenge delete de caps/settlement ZSET; TTL cego em `settlement:queue:priority`; reintroduzir pandas; inchacar allowlist vulture com codigo de negocio.

## Entrega

Lista do que saiu + o que ficou de proposito + hashes de commit por onda.

Apos mudanca material (mesmo sem purge): skill `aether-surface-sync`.
