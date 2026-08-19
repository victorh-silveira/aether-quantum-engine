## [2.14.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.13.0...v2.14.0) (2026-08-19)

### Funcionalidades

* **infra:** refatora modelos e fit adaptativo dos microservicos ml ([3335dd5](https://github.com/victorh-silveira/aether-quantum-engine/commit/3335dd5f24e1935d25479cb2224670fa4867e5fb))

## [2.13.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.12.2...v2.13.0) (2026-08-19)

### Funcionalidades

* **app:** migra para rotulagem supertrend_atr e treino focado 5m ([62b3c59](https://github.com/victorh-silveira/aether-quantum-engine/commit/62b3c59a06671a3ab2734ab653d365c9519d38a1))

## [2.12.2](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.12.1...v2.12.2) (2026-08-19)

### Correcoes de Bug

* **engine:** exige corpo minimo da vela no anti-loss ([f8985b5](https://github.com/victorh-silveira/aether-quantum-engine/commit/f8985b5419498e40dfc6b195b9935b2f9f6f4d6e))
* **orchestrator:** estabiliza treino, gates de execucao e makefile ([1695670](https://github.com/victorh-silveira/aether-quantum-engine/commit/1695670d2e424245b4d548cbfe559831db2271a9))
* **release:** fixa versao do conventionalcommits para compatibilidade com sem-rel ([a5264e2](https://github.com/victorh-silveira/aether-quantum-engine/commit/a5264e2692be49b1548a6eedc9cd2caa77e6f08d))

## [2.12.1](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.12.0...v2.12.1) (2026-08-17)

## [2.12.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.11.0...v2.12.0) (2026-08-17)

## [2.11.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.10.0...v2.11.0) (2026-08-17)

## [2.10.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.9.0...v2.10.0) (2026-08-16)

## [2.9.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.8.6...v2.9.0) (2026-08-16)

## [2.8.6](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.8.5...v2.8.6) (2026-08-14)

## [2.8.5](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.8.4...v2.8.5) (2026-08-14)

### Removed
* **infra:** servidor de inferencia gRPC removido da stack Docker; TCN = eager/CUDA local; profiles `core`/`ml` apenas

## [2.8.4](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.8.3...v2.8.4) (2026-08-13)

## [2.8.3](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.8.2...v2.8.3) (2026-08-13)

## [2.8.2](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.8.1...v2.8.2) (2026-08-11)

## [2.8.1](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.8.0...v2.8.1) (2026-08-11)

## [2.8.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.7.0...v2.8.0) (2026-08-10)

## [2.7.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.6.1...v2.7.0) (2026-08-10)

## [2.6.1](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.6.0...v2.6.1) (2026-08-10)

## [2.6.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.5.0...v2.6.0) (2026-08-10)

## [2.5.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.4.0...v2.5.0) (2026-08-10)

## [2.4.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.3.0...v2.4.0) (2026-08-10)

## [2.3.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.2.0...v2.3.0) (2026-08-10)

## [2.2.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.1.0...v2.2.0) (2026-08-10)

## [2.1.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.0.0...v2.1.0) (2026-08-09)

## [2.0.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.82.0...v2.0.0) (2026-08-08)
* **loss_clf_calibrate:** floor FLIP **0.90**; fit `class_weight=balanced` + `min_child_samples=15`; retrain pos-LOSS bloqueado se buffer LOSS-heavy (`loss/n>0.60` ou `win<8`) exceto saida bootstrap (≥1 WIN+≥1 LOSS); vetor 24D clip edge idx10 + `micro_tick_acceleration` idx19; Prob/Cal e `p_loss` com **5** casas; limpeza+bootstrap embutidos em `make docker-rebuild` / `docker-reset` (`COLD_START`, `p_loss=0.50`, veto off ate retrain real)
* **scale_mili_tape_chop:** `adapt_mili_tape_skip_chop` **true** — nao inverter TCN so com mili+tape em micro=chop (majority com lead permanece)
* **otc_spc/m15:** contrato/OHLC M15; ciclo/assinatura **60 s** (entrada a cada 1 m); payout live **0.72**; settle tolerancia **300 s**; watchdog stale **600 s**; SIDE_EQ/sample_size densos; price_zone off
* **loss_clf_flip:** `hard_p_loss_floor` **0.90** → **FLIP** CALL↔PUT **ancorado no TCN** so com `veto_ready`; bootstrap devolve `p_loss=0.50` (sem FLIP); soft Kelly em `[0.65, 0.90)` com `veto_ready`; saida bootstrap no `/learn` com ≥1 WIN+≥1 LOSS
* **loss_clf_hard:** mandato **2026-08-07** — substituido por **loss_clf_flip** (floor **0.90**)
* **loss_clf_flip_legacy:** floor **0.80** + FLIP no bootstrap removidos apos anti-TCN em cold-start
* **regime_chop:** mandato **2026-08-07** — pausa M15 se ADX &lt; **0.22** e Hurst ∈ [**0.47**, **0.53**] (`SKIP:REGIME_CHOP`)
* **scale_majority:** `adapt_on_majority_votes` conta TCN/tape/mili/mini_pair/RSI; lado com mais votos adapta EXEC
* **stake_2pct:** explore piso **2%** banca (`neutral_bankroll_pct`/`min_stake_pct`); loss_clf soft nao esmaga U; RECOVER `cover_multiple` **2.0** (loss+win); tetos linear2/3 **5%**
* **remove_calib_gray:** remove `hold_calib_gray` / `hold_cal_margin` / `adapt_*_cal_margin` / `calib_gray_*` soft Kelly+teto / log `CALIB_GRAY`
* **risk:** piso explore **1%** / teto **5%**; loss-clf soft_max **2%** / soft_mult_high **0.40**
* **startup:** remove `startup_fetch_bars: 512` (era &lt; lookback 720 → SKIP data); piso = `min_dl_inference_len`; sync lean tambem em inferencia
* **infer:** piso `inference_history_bars` sem double-count de lookback (~864 vs ~1456); evita SKIP:DATA quando API R_10 entrega ~1.2k M15
* **risk:** pos-LOSS sem flip DIR_LOCK (lado fica no TCN); cover RECOVER sem damping de stop-win; loss-clf soft waivado com pending
