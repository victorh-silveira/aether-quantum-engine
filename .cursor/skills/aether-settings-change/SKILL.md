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
6. Nao afrouxar doutrina de lado/sizing: `flip_*` (ex.: `flip_require_auto_learn`, `flip_seed_*`, `flip_waive_tcn_pos_edge_on_discord`), `fusion_*` (ex.: `fusion_block_when_tcn_pos_edge`, `fusion_block_when_tcn_candle_agree`, `fusion_loss_requires_auto_learn`, soft **0.50/0.25**, `fusion_meta_ev_weight` **0.0**, `fusion_loss_weight` **0.0**, `fusion_loss_seed_weight_mult` **0.0**), `ops_window_bars` **3**, `neg_edge_*` (ex.: `neg_edge_deep_edge_floor` **−0.12**), `anti_loss_hard_skip` (**true**), `anti_loss_min_candle_body` (**0.10**), `anti_loss_live_weak_candle_enabled` (**false**), `anti_loss_live_confirm_enabled` (**false**), `anti_loss_live_confirm_min_body` (**0.15**), `anti_loss_live_exec_candle_enabled` (**false**), `anti_loss_allow_candle_flip` (**false**), `anti_loss_seed_discord_enabled` (**false**), `regime_gate_enabled` (**true**), RSI knobs legado **0.30/0.70**, `invert_exec_side` (**false**), `online_training` (**false**), `cover_enabled` **false**, piso Kelly **1%**, Soft_SIZE com PEND preserva piso elevado, caps recovery linear3 **2.5%**
7. `kelly_p_floor` e piso de **p**; com `fusion_applied`, Kelly ancora em `fusion_p_eff` **so se** Cal TCN ja passou `neg_edge`; `neutral_bankroll_pct` / `min_stake_pct` **0.01** (piso operacional 1%; nao broker `stake_min`); Soft_SIZE eleva a `soft_size_min_stake_pct` **0.025** so se Edge >= `soft_size_min_edge` **0.015**; `explore_stake_scale_floor` **0.40**; `target_damping_floor`/`span` **0.50**/**0.50** (inicio damping **1.0**, perto-meta **0.50**); com `cover_enabled` false PEND nao usa cover para tamanho
8. Nao reintroduzir `kelly_no_edge` como pause de sizing; nao reintroduzir `hard_cal_margin_floor` / quality gate amplo

## Proibido

Hardcode do mesmo numero em codigo + settings + teste divergentes.
Revenge sizing / subir caps linear “para aprender” com N baixo.

Doc: `docs/engineering-settings-ssot.md`

- TCN ortogonal **14D** / meta **23D**; limiares **0.53/0.47**; `ema_50` **50**; anti-loss direcional off; `regime_gate_enabled` HARD squeeze.
