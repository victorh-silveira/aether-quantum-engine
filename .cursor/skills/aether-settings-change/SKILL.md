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
6. Loss-clf: `veto_mode` **hard**; mature **0.58**; young **0.58**; FLIP so apos auto_learn; `bootstrap_exit_n` **4**; `flip_trust_n` **32**; `ready_n` **32**; `retrain_min_n` **12**; **sem** `candle_holds`; **proibido** Soft Kelly / HARD SKIP no piso; `invert_exec_side` **false**; `skip_neg_edge` **true**; `skip_exec_vs_candle` / `skip_scale_candle_discord` / `skip_doji` **false**; `skip_below_soft_min_acc` **false**
7. Sizing: `cover_enabled` **true**, amort **1/1**, stake = `min(max(PEND/payout, 1% banca), cap_L0)`; cover L0 **3.5%**; PEND nao force-explore por near-stop; piso Kelly **1%** soberano, Single-Strike 4.31% so com `live_n >= 12` e conviction ≥ **0.58**; `online_training` **false**; cooldown pos-LOSS LIN>=1 (300/300/600/900s); pausa 3/5/2
8. Edge CLUSTER = EV payout **0.85**; **SKIP** se ≤ 0 em EXPLORE (`skip_neg_edge`); waived com PEND material; Edge>0 exige margem Cal ≳ **0.041**
9. Calibracao: `min_oos_sharpness` / `min_calibration_sharpness` **0.0** (export sem piso); `min_calibration_margin_floor` **0.05** (live sem stretch); deploy `force_ok` **true**
10. SCALE: `adapt_retract_enabled` **false**
11. META: LEARN/telemetria; soft Kelly **inerte**
12. Nao reintroduzir quality gate amplo / SCALE adapt de lado / Soft Kelly META; nao desligar `skip_neg_edge`

## Proibido

Hardcode do mesmo numero em codigo + settings + teste divergentes.
Revenge sizing / subir caps linear “para aprender” com N baixo.

Doc: `docs/engineering-settings-ssot.md` + `docs/engineering-indicator-gates.md`
