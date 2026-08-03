# Deep Learning e meta

Guia operacional DL para agentes. Detalhe de features: [`arquitetura.md`](arquitetura.md) §4–5.

## Runtime atual (SSOT settings)

| Item | Valor tipico |
|------|----------------|
| Simbolo | `R_10` |
| Arch | TCN |
| Lookback | **360** → tensor `[1, 360, 34]` |
| Macro | **600 s** |
| Micro / contrato | **120 s** |
| Features | **34D** (`FEATURE_DIM`) |
| Label | `spot_forward` (1 barra micro = contrato 120s) |
| ACC / deploy | `soft_min_val_accuracy` **0.53**; `deploy_gate.enabled=true`, `force_ok=false` |
| Early stop | `min_epochs` **40**, `early_stopping_patience` **25** |
| Meta | LightGBM **43D** `predicted_payoff_edge` |

## Entry points

| Comando | Papel |
|---------|-------|
| `train.py` / `app/train.py` | treino TCN |
| `app/scripts/batch/launch-train.bat` | DL → gate ACC/deploy → Timescale → meta |
| `app/scripts/operations/check_dl_deploy_gate.py` | aborta meta se ACC&lt;0.53 ou `deploy_ok=false` |
| `app/scripts/operations/train_meta_*.py` | treino offline do meta (`--source auto`) |

## Deploy gate (senior)

`resolve_deploy_ok` exige `val_accuracy >= soft_min` **antes** de `mini_ok` / `force_ok`. Checkpoint com ACC 0.52 grava `deploy_ok=false`. Nao usar `force_ok=true` nem `bypass_deploy_gate=true` em producao.

## Meta — alvo e dados

- Alvo preferencial: z-score do forward return; se closes/fwd flat → payoff assinado (`_continuous_payoff_target`).
- Timescale curto/flat (hydrate sintetico) e rejeitado; fallback Deriv com piso ≥ **2000** barras.
- `validate_target_variance` inclui `source`, `forward_var`, `close_nunique`.

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
