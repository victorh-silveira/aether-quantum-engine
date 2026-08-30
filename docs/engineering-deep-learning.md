# Deep Learning e meta

Guia operacional DL para agentes. Detalhe de features: [`arquitetura.md`](arquitetura.md) §4–5.

## Runtime atual (SSOT settings)

| Item | Valor tipico |
|------|----------------|
| Simbolo | **1HZ75V** (Volatility 75 (1s)) |
| Arch | TCN |
| Lookback | **30** → tensor `[1, 30, 34]` |
| MACRO OHLC | **86400 s** (D1, 365 velas de treino, `data_handler.granularity`) |
| MICRO OHLC | **900 s** (M15, 500 velas, `data_handler.micro_granularity`) |
| Contrato | **15 m** RISE_FALL (ops fixo M15); label TCN **N=1** vela M15 (`supertrend_atr`) |
| MINI OHLC | **900 s** (`mini_granularity`) |
| Bootstrap wait | `bootstrap_history_wait_cap_seconds` **30** (nao dorme a granularidade inteira entre retries) |
| MILI | Tick flow (nao OHLC) |
| Features | **34D** (`FEATURE_DIM`) |
| Label | `triple_barrier` (Triple Barrier Method: Log-Vol Barriers + Expiry) |
| Online training | **false** (DEMO usa checkpoint do `launch-train`) |
| ACC / deploy | `soft_min_val_accuracy` **0.53**; `max_brier` / `soft_max_brier` **0.28**; `force_ok=false` |
| Retries | `train_deploy_retries` **6** (reseed + reset de pesos) |
| Early stop | `min_epochs` **15**, `early_stopping_patience` **17** |
| Meta | LightGBM **43D** `predicted_payoff_edge` |

## Entry points

| Comando | Papel |
|---------|-------|
| `train.py` / `app/train.py` | treino TCN |
| `app/scripts/batch/launch-train.bat` | sanitize → sweep horizonte N (H15–H60 em M1) + promote → gate → Timescale → meta |
| `app/scripts/operations/run_launch_train_tf_pipeline.py` | orquestra sweep horizonte N + promote (fallback `train.py` se `horizon_sweep.run_in_launch_train=false`) |
| `make docker-rebuild` | recarrega meta/loss apos o treino (**nao** apaga `data/dl`) |
| `app/scripts/operations/sanitize_fresh_run.py` | limpa `data/dl`, meta/loss pkls e estado em `data/` (so train/reset) |
| `app/scripts/operations/check_dl_deploy_gate.py` | aborta meta se geometria invalida; aceita **settle_wr** elegivel (mesmos knobs do sweep) ou ACC≥0.53 + soft path; simbolos de `settings.symbols` |
| `app/scripts/operations/train_meta_*.py` | treino offline do meta (`--source auto`; simbolos de settings) |
| `app/scripts/operations/sweep_train_timeframes.py` | loop de celulas H15–H60; artefactos em `data/dl/sweep/R_10/H{N}`; leaderboard JSON |
| `app/scripts/operations/promote_tf_winner.py` | promove vencedor elegivel para `settings.json` + `drift_symbols.py` + `data/dl` (fail-closed se nenhum) |

## Sweep de horizonte N (launch-train)

O TCN estima deslocamento em **N velas M1**. O `launch-train` treina a grade **H15–H60** (`duration_minutes`/`n_bars` = 15/20/…/60 em M1 — **nao** TF M15/900s; treino por celula `duration=N` alinhado ao label), loga **uma** linha `[HORIZON] cell i/N …` por celula (com `why=` se deploy=0) e pos-sweep denso (`board candidatos/elegiveis/skip`, winner, promote com paths relativos; Timescale `--check-only` e `[META] ok` em 1 linha cada; sem dump JSON/params), e promove o mais assertivo (`settle_wr` ≥ be+0.03, n≥16, history≥800). Artefactos em `data/dl/sweep/R_10/H{N}/`. **SSOT atual** (ultimo promote): `label_horizon_bars` **45** (H45); contrato ops **5 m** (fixo; promote **nao** exporta duration).

Pipeline **offline** (nao troca N por ciclo ao vivo):

1. `horizon_sweep.duration_minutes` (ou `n_bars`) no SSOT — celulas **H15…H60** no relogio M1 (lookback/history copiados, sem reescalar wall-clock). Simbolo **R_10**.
2. `run_launch_train_tf_pipeline.py` limpa `data/dl/sweep`, treina cada celula com ckpt isolado (`data/dl/sweep/R_10/H{N}/`), **infra/MinIO off** no sweep, **1 tentativa** (`train_deploy_retries=1`), overlay `logging.level=CRITICAL` se `quiet_train_logs` (falha resumida em `why=` na linha cell), grava leaderboard.
3. Elegivel: **`settle_wr` ≥ be + 0.03** **e** `settle_n ≥ min_settle_n` (**16**) **e** `history_bars ≥ min_history_bars` (**800**). Label ACC e so telemetria.
4. Com `auto_promote=true` (default), promove vencedor: copia ckpt para `data/dl/` (carimba `deploy_ok`) + grava `label_horizon_bars` do winner; **`params.duration`** vem de `ops_contract_duration_minutes` (**5**), **nao** do N do winner — **fail-closed** se nenhum elegivel (meta nao roda).
5. Gate ACC/settle + meta no SSOT promovido. Depois: `make docker-rebuild` + sync MinIO.

Knobs: `horizon_sweep.duration_minutes` / `n_bars` / `ops_contract_duration_minutes` / `quiet_train_logs` / `run_in_launch_train` / pisos settle. Flags CLI: `--only H15 H30`, `--dry-run`, `--skip-promote`.

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

`resolve_deploy_ok` (treino) exige `val_accuracy >= soft_min` **antes** de `mini_ok` / `force_ok`. Checkpoint com ACC 0.52 grava `deploy_ok=false` no treino, mas o sweep ainda persiste settle_* para ranking. Pos-promote, `check_dl_deploy_gate` **e** o load DEMO (`_effective_deploy_ok`) aceitam o ckpt se **settle_wr** passar a elegibilidade do sweep; senao caem no path ACC/Brier/collapse. Sem isso, promote M1 com ACC&lt;0.53 gera `SKIP:DEPLOY` eterno na DEMO. Treino rejeitado **sempre sobrescreve** o `.pth` anterior (sem preservar deploy antigo). Nao usar `force_ok=true` nem `bypass_deploy_gate=true` em producao.

Majority-collapse (alem do ACC): com `reject_majority_collapse=true`, rejeita se a fracao predita colapsa (`|pred_call_frac-0.5|` ou `|pred_call_frac-label_call_frac|` &gt; `max_label_call_frac_bias` **0.20**) — mesmo com `minority_recall` acima do piso (ex.: labels ~0.5 e `pred_call=0.24`). Alternativa: label viesado (`|label_call_frac-0.5|` &gt; bias) **e** `minority_recall < min_minority_recall` (**0.25**). Checkpoint de treino nao promove pico com `collapse_hit`.

Checkpoint de treino restaura o melhor estado **sharp** por **maior val_acc** apenas se a **BCE** de validacao &lt; **0.70** (abaixo de chute aleatorio; pico da epoca 1 com loss 0.80 e ignorado). `focal_gamma` afeta so o treino — nao o monitor de checkpoint. Loss so desempata. Em R_10, `spot_forward` costuma platô ~0.52; `ma_trend` e o label SSOT atual.

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

Live: apos `apply_calibration_neutral_tolerance`, `clamp_calibrated_call_to_raw_band` clipa **p_call** em `[raw±max_calibrated_raw_gap]` (**0.08**); PUT espelha `1-p_call`. Metricas: `cal_raw_gap_capped` / `cal_raw_gap`. Isso alimenta Edge/FUSION/Kelly (nao so `trade_score`). `temperature_min` **1.0** impede T&lt;1 no fit.

Fusao: `why=tcn_pos_edge` exige Cal **e** raw_edge ≥ `fusion_min_edge_execute` (**0.04**). Sintoma de regressao: CLUSTER Prob≈BE + `p_put`≫0.70 + `why=tcn_pos_edge` com `raw_edge`~0.

Visao multi-escala (MACRO/MICRO/MINI/MILI) e soft Kelly ficam fora do pacote DL — ver [`engineering-orchestrator.md`](engineering-orchestrator.md) e `orchestrator.execution.scale_vision`.

## Pos-migrate hibrido (legado → SSOT atual M1)

1. Invalidar checkpoints `data/dl/*.pth` e TorchScript MinIO com `granularity`/`lookback` ≠ settings (ex.: legado **180**/7200 M3, **120**/3600 ou lookback **720**).
2. Re-hydrate Timescale (`docker-hydrate.sh` / `ensure_timescale`) para OHLC micro/MINI **60** / macro **7200**.
3. Retreinar com **`launch-train.bat`** (TCN `lookback=480`, micro **60**, label N promovido; contrato ops **5 m**) + meta — **nao** via `launch-all-demo`.
4. So depois: `launch-all-demo.bat`; validar CFG live `ohlc=60s`, `macro=7200s`, `contrato=5 m`, `label_horizon_bars=55`.

Com `online_training=false` (SSOT), a DEMO nao agenda retreino TCN em runtime (nem settle nem rolling); usa o checkpoint do `launch-train`. Para reativar, `online_training=true` + `rolling_retrain_bars` / `retrain_min_bars` (sem `mark_force_retrain` no settle). Meta e loss-clf fazem `/v1/learn` a cada trade.

## Pacote

`app/src/application/services/deep_learning/` — features, labels, predict, calibracao, deploy, checkpoint.

## Meta runtime

- Inferencia TCN: eager/CUDA local no host (`data/dl`)
- Meta HTTP: `aether-meta-classifier`; artefato em `infra/docker/meta-models/`

## Anti-padroes

- Trocar `label_mode` sem retreinar
- `deploy_gate.force_ok=true` ou gate desligado
- Treinar meta em OHLC sintetico/flat do hydrate Docker
- Seguir `launch-train` para meta com ACC&lt;0.53

Skill: `aether-dl-train`.
