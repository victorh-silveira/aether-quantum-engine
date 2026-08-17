---
name: aether-settings-change
description: >-
  Altera knobs em config/settings.json com SSOT fail-closed (resolve_*, testes,
  docs). Use when changing settings.json, quality_gate, Kelly, sample_size_policy,
  ACC/edge floors, or any runtime knob.
---

# Mudanca de settings

## Checklist

1. Identificar bloco SSOT (`data_handler`, `deep_learning`, `orchestrator.execution`, `risk_management`, `infra`)
2. Default seguro; parsers `resolve_*` / config tipada
3. Teste do default e do branch novo (`test_doctrine_settings_ssot` quando knob de doutrina)
4. Atualizar doc se semantica operacional mudar (`engineering-settings-ssot`, medallion, doutrina, `AGENTS.md`)
5. Nao afrouxar sem mandato + evidencia de frequencia: `force_trade`, edge floor, ACC gate, `max_safe_stake_*`, `sample_size_policy` (**12/32**)
6. Nao afrouxar doutrina de lado/sizing: `flip_*` (ex.: `flip_require_auto_learn`, `flip_seed_*`), `fusion_*` (ex.: `fusion_block_when_tcn_pos_edge`, `fusion_block_when_tcn_candle_agree`, `fusion_loss_requires_auto_learn`, soft **0.40/0.25**, `fusion_loss_seed_weight_mult` **0.0**), `neg_edge_*` (ex.: `neg_edge_deep_edge_floor` **−0.12**), `anti_loss_hard_skip` (**true**), `anti_loss_min_candle_body` (**0.10**), `invert_exec_side` (**false**), `online_training` (**false**), caps recovery (amort **1/1**, cover **1.50**, linear3 **2.5%**)
7. `kelly_p_floor` e piso de **p**; com `fusion_applied`, Kelly ancora em `fusion_p_eff`; `neutral_bankroll_pct` e piso operacional de stake explore (nao broker `stake_min`); `explore_stake_scale_floor` **0.40**; `target_damping_floor`/`span` **0.50**/**0.50** (inicio damping **1.0**, perto-meta **0.50**); RECOVER nao usa `bankroll×f*` para tamanho
8. Nao reintroduzir `kelly_no_edge` como pause de sizing; nao reintroduzir `hard_cal_margin_floor` / quality gate amplo

## Proibido

Hardcode do mesmo numero em codigo + settings + teste divergentes.
Revenge sizing / subir caps linear “para aprender” com N baixo.

Doc: `docs/engineering-settings-ssot.md`
