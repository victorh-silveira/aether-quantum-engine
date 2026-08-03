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
| Label | `spot_forward` (alt: `ma_trend`, Triple Barrier) |
| Thresholds | CALL/PUT **0.51/0.49**; neutro off |
| Meta | LightGBM **43D** `predicted_payoff_edge` (opcional para execucao) |

## Entry points

| Comando | Papel |
|---------|-------|
| `train.py` / `app/train.py` | treino de sessao / bootstrap |
| `run.py` / `app/run.py` | operacao (inferencia) |
| `app/scripts/operations/train_meta_*.py` | treino offline do meta |

## Pacote

`app/src/application/services/deep_learning/` — features, labels, predict, calibração, deploy, checkpoint, device/CUDA.

Checkpoint deve bater **lookback** e **granularity** esperados (`dl_startup` / `dl_symbol_runtime`).

## Triton / meta

- Triton gRPC: opcional nos settings (`infra.triton.enabled` / `require_for_execution`)
- Meta HTTP: `aether-meta-classifier`; `require_meta_for_execution: false` nos settings atuais
- Fail-closed so se reativado explicitamente

## Anti-padroes

- Trocar `label_mode` sem retreino e sem hipotese
- Ignorar `val_accuracy` / ACC gate
- Assumir que meta sempre esta ligado

Skill: `aether-dl-train`.
