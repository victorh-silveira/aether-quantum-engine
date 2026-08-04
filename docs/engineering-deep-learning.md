# Deep Learning e meta

Guia operacional DL para agentes. Detalhe de features: [`arquitetura.md`](arquitetura.md) §4–5.

## Runtime atual (SSOT settings)

| Item | Valor tipico |
|------|----------------|
| Simbolo | `R_10` |
| Arch | TCN |
| Lookback | **360** → tensor `[1, 360, 34]` |
| MACRO OHLC | **600 s** (`data_handler.granularity`) |
| MICRO / contrato (TCN) | **120 s** (`micro_granularity`) |
| MINI OHLC | **60 s** (`mini_granularity`) |
| MILI | Tick flow (nao OHLC) |
| Features | **34D** (`FEATURE_DIM`) |
| Label | `ma_trend` (MA 8; horizonte 1 barra micro = contrato 120s) |
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

Majority-collapse (alem do ACC): com `reject_majority_collapse=true`, rejeita se `|label_call_frac - 0.5| > max_label_call_frac_bias` (**0.20**) e `minority_recall < min_minority_recall` (**0.25**).

Checkpoint de treino restaura pesos do **melhor val_acc** (melhoria so de loss nao sobrescreve). Em R_10, `spot_forward` costuma platô ~0.52; `ma_trend` e o label SSOT atual.

## Meta — alvo e dados


- Alvo preferencial: z-score do forward return; se closes/fwd flat → payoff assinado (`_continuous_payoff_target`).
- Timescale curto/flat (hydrate sintetico) e rejeitado; fallback Deriv com piso ≥ **2000** barras.
- `validate_target_variance` inclui `source`, `forward_var`, `close_nunique`.

## Calibracao: `raw_extreme` (anti-override)

Modo legado `tcn_macro_override` foi substituido por `raw_extreme` em `dl_calibration_tolerance.py`:

- Se `raw` > `tcn_macro_call_override` ou `raw` < `tcn_macro_put_override`, o modo vira `raw_extreme`.
- **Cal nao e substituido por raw**; retorno mantem probabilidade calibrada.
- Kelly / sizing usam **Cal**, nao raw.
- Nomes das chaves SSOT (`tcn_macro_*_override`) sao historicos: limiam extremo de **raw TCN**, nao o timeframe MACRO OHLC.

Visao multi-escala (MACRO/MICRO/MINI/MILI) e soft Kelly ficam fora do pacote DL — ver [`engineering-orchestrator.md`](engineering-orchestrator.md) e `orchestrator.execution.scale_vision`.

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
