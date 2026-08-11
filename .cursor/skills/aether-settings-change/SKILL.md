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
3. Teste do default e do branch novo (`test_doctrine_settings_ssot` quando knob de doutrina)
4. Atualizar doc se semantica operacional mudar (`engineering-settings-ssot`, medallion, doutrina, `AGENTS.md`)
5. Nao afrouxar sem mandato + evidencia de frequencia: `force_trade`, cal floor, edge floor, ACC gate, `max_safe_stake_*`, `sample_size_policy`
6. Nao afrouxar doutrina de lado/sizing: `flip_*` (ex.: `flip_require_auto_learn`, `flip_seed_*`), `fusion_*` (ex.: `fusion_block_when_tcn_pos_edge`, soft **0.40/0.25**), `neg_edge_*` (ex.: `neg_edge_deep_edge_floor` **−0.12**), `invert_exec_side` (**false**), `online_training` (**false**), caps recovery (amort **4–6**, cover **1.25**, linear3 **2.5%**)
7. `kelly_p_floor` e piso de **p**; `neutral_bankroll_pct` e piso operacional de stake explore (nao broker `stake_min`)
8. Nao reintroduzir `kelly_no_edge` como pause de sizing

## Proibido

Hardcode do mesmo numero em codigo + settings + teste divergentes.
Revenge sizing / subir caps linear “para aprender” com N baixo.

Doc: `docs/engineering-settings-ssot.md`
