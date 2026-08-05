---
name: aether-settings-change
description: >-
  Altera knobs em config/settings.json com SSOT fail-closed (resolve_*, testes,
  docs). Use when changing settings.json, quality_gate, Kelly, sample_size_policy,
  hard_cal_margin_floor, or any runtime knob.
---

# Mudanca de settings

## Checklist

1. Identificar bloco SSOT (`data_handler`, `deep_learning`, `orchestrator.execution`, `risk_management`, `infra`)
2. Default seguro; parsers `resolve_*` / config tipada
3. Teste do default e do branch novo
4. Atualizar doc se semantica operacional mudar (`engineering-settings-ssot`, medallion, doutrina)
5. Nao afrouxar: force_trade, cal floor, edge floor, ACC gate, max_safe_stake_*, sample_size_policy sem mandato
6. `kelly_p_floor` e piso de **p**; `neutral_bankroll_pct` e piso operacional de stake explore (nao broker `stake_min`)
7. Nao reintroduzir `kelly_no_edge` como pause de sizing

## Proibido

Hardcode do mesmo numero em codigo + settings + teste divergentes.

Doc: `docs/engineering-settings-ssot.md`
