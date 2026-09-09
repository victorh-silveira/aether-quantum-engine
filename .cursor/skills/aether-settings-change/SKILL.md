---
name: aether-settings-change
description: >-
  Altera knobs em config/settings.json com SSOT fail-closed (resolve_*, testes,
  docs). Use when changing settings.json, Kelly, sample_size_policy,
  ACC/edge floors, loss-clf FLIP, or any runtime knob.
---

# Mudanca de settings

## Checklist

1. Identificar bloco SSOT (`data_handler`, `deep_learning`, `orchestrator.execution`, `risk_management`, `infra`)
2. Default seguro; parsers `resolve_*` / config tipada
3. Teste do default e do branch novo (`test_doctrine_settings_ssot` quando knob de doutrina)
4. Atualizar doc se semantica operacional mudar (`engineering-settings-ssot`, doutrina, `AGENTS.md`)
5. Nao afrouxar sem mandato + evidencia: `force_trade`, ACC gate, `max_safe_stake_*`, `sample_size_policy` (**12/32**)
6. Loss-clf: `veto_mode` **hard**; piso **0.55** (`hard_p_loss_floor` + `flip_young_p_eff_floor`); FLIP so apos auto_learn; `flip_trust_n` **32**; `ready_n` **32**; `retrain_min_n` **12**; tape so telemetria; **proibido** Soft Kelly / HARD SKIP no piso / bloco `signal_skip` / `invert_exec_side` / fusao de lado
7. Sizing: `cover_enabled` **false**, piso Kelly **1%**, Single-Strike 4.31%; `online_training` **false**
8. Nao reintroduzir quality gate amplo / price_zone / `hard_cal_margin_floor`

## Proibido

Hardcode do mesmo numero em codigo + settings + teste divergentes.
Revenge sizing / subir caps linear “para aprender” com N baixo.

Doc: `docs/engineering-settings-ssot.md` + `docs/engineering-indicator-gates.md`
