# Deep Learning e meta

Guia operacional DL para agentes. Detalhe de features: [`arquitetura.md`](arquitetura.md) §4–5.

## Runtime atual (SSOT settings)

| Item | Valor tipico |
|------|----------------|
| Simbolo | `R_10` |
| Arch | TCN |
| Lookback | **720** → tensor `[1, 720, 34]` (~7,5 dias @ 900 s) |
| MACRO OHLC | **3600 s** (`data_handler.granularity`) |
| MICRO (TCN) | **900 s** (`micro_granularity`) — M15 |
| Contrato | **15 m** RISE_FALL (label 1 barra micro = 900 s) |
| MINI OHLC | **900 s** (`mini_granularity`) |
| Bootstrap wait | `bootstrap_history_wait_cap_seconds` **30** (nao dorme 900 s entre retries) |
| MILI | Tick flow (nao OHLC) |
| Features | **34D** (`FEATURE_DIM`) |
| Label | `ma_trend` (MA 8; horizonte 1 barra micro **900 s**; alinhado ao contrato M15) |
| ACC / deploy | `soft_min_val_accuracy` **0.53**; `soft_max_brier` **0.26**; `force_ok=false` |
| Early stop | `min_epochs` **40**, `early_stopping_patience` **40**; restore **best val_acc** |
| Meta | LightGBM **43D** `predicted_payoff_edge` |

## Entry points

| Comando | Papel |
|---------|-------|
| `train.py` / `app/train.py` | treino TCN |
| `app/scripts/batch/launch-train.bat` | DL → gate ACC/deploy → Timescale → meta |
| `app/scripts/operations/check_dl_deploy_gate.py` | aborta meta se ACC&lt;0.53 ou `deploy_ok=false` |
| `app/scripts/operations/train_meta_*.py` | treino offline do meta (`--source auto`) |

## Sample weighting (vies de classe + dinamica)

SSOT: `deep_learning.sample_weighting` — modulo `dl_sample_weighting.py`; wired em `train_model_walkforward` via `compose_train_weights` (alinha pesos do purged-split).

| Knob | Padrao | Papel |
|------|--------|-------|
| `class_balance_enabled` | true | Repondera CALL/PUT quando a taxa de label sai do equilibrio |
| `class_balance_eps` | 0.05 | Tolerancia antes de balancear |
| `recency_enabled` | true | Decai peso de amostras antigas |
| `recency_half_life_n` | **2000** | Half-life em numero de amostras |

Telemetria de treino: `TrainResult.label_call_frac`, `pred_call_frac`, `minority_recall` (logados em treino bem-sucedido).

## Deploy gate (senior)

`resolve_deploy_ok` exige `val_accuracy >= soft_min` **antes** de `mini_ok` / `force_ok`. Checkpoint com ACC 0.52 grava `deploy_ok=false`. Nao usar `force_ok=true` nem `bypass_deploy_gate=true` em producao.

Majority-collapse (alem do ACC): com `reject_majority_collapse=true`, rejeita se `minority_recall < min_minority_recall` (**0.25**) e houver vies (`|pred_call_frac-0.5|`, `|pred_call_frac-label_call_frac|` ou `|label_call_frac-0.5|` &gt; `max_label_call_frac_bias` **0.20`). Labels balanceados com `pred_call` colapsado (ex. 0.83) tambem falham o gate.

Checkpoint de treino restaura pesos do **melhor val_acc** (melhoria so de loss nao sobrescreve). Em R_10, `spot_forward` costuma platô ~0.52; `ma_trend` e o label SSOT atual.

## Meta — alvo e dados


- Alvo preferencial: z-score do forward return; se closes/fwd flat → payoff assinado (`_continuous_payoff_target`).
- Timescale curto/flat (hydrate sintetico) e rejeitado; fallback Deriv com piso ≥ **2000** barras.
- `validate_target_variance` inclui `source`, `forward_var`, `close_nunique`.

## Calibracao: `raw_extreme` (anti-override)

Modo legado `tcn_macro_override` foi substituido por `raw_extreme` em `dl_calibration_tolerance.py`:

- Se `raw` > `tcn_macro_call_override` ou `raw` < `tcn_macro_put_override`, o modo vira `raw_extreme`.
- **Cal nao e substituido por raw**; retorno mantem probabilidade calibrada.
- Com Cal na banda neutra (`calibration_neutral_drift` **[0.47, 0.53]**; drift degenerado `[0.5,0.5]` rejeitado), o **lado** segue o limiar raw (`raw_dir`), nao Cal≥0.5.
- Kelly / sizing usam **Cal**, nao raw.
- Nomes das chaves SSOT (`tcn_macro_*_override`) sao historicos: limiam extremo de **raw TCN**, nao o timeframe MACRO OHLC.

`calibration.method=auto` + piso `min_calibration_sharpness` / `min_oos_sharpness` (**0.01**): se temperatura/Platt/isotonico colapsar nitidez, o fit cai para `identity` (raw). Export mede sharpness via `apply_calibrator_stable` (mesmo caminho do live).

Visao multi-escala (MACRO/MICRO/MINI/MILI) e soft Kelly ficam fora do pacote DL — ver [`engineering-orchestrator.md`](engineering-orchestrator.md) e `orchestrator.execution.scale_vision`.

## Pos-migrate hibrido 30s/60s

1. Invalidar checkpoints `data/dl/*.pth`, TorchScript MinIO/Triton com `granularity=120`.
2. Re-hydrate Timescale (`docker-hydrate.sh` / `ensure_timescale`) para OHLC **60** / **300**.
3. Retreinar com **`launch-train.bat`** (TCN `lookback=720`, micro 60) + meta — **nao** via `launch-all-demo`.
4. So depois: `launch-all-demo.bat`; validar CFG live `ohlc=60s`, `macro=300s`, `contrato=30s`.

Com `online_training=false`, DEMO/`launch-all-demo` **nunca** agenda bootstrap; checkpoint ausente → warning + SKIP tecnico ate treino offline.

## Pacote

`app/src/application/services/deep_learning/` — features, labels, predict, calibracao, deploy, checkpoint.

## Triton / meta runtime

- Triton gRPC: opcional (`infra.triton.enabled`)
- Meta HTTP: `aether-meta-classifier`; artefato em `infra/docker/meta-models/`

## Anti-padroes

- Trocar `label_mode` sem retreinar
- `deploy_gate.force_ok=true` ou gate desligado
- Treinar meta em OHLC sintetico/flat do hydrate Docker
- Seguir `launch-train` para meta com ACC&lt;0.53

Skill: `aether-dl-train`.
