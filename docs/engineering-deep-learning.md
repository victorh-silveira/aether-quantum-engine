# Deep Learning e meta

Guia operacional DL para agentes. Detalhe de features: [`arquitetura.md`](arquitetura.md) §4–5.

## Runtime atual (SSOT settings)

| Item | Valor tipico |
|------|----------------|
| Simbolo | **R_10** (Volatility 10) |
| Arch | TCN |
| Lookback | **480** → tensor `[1, 480, 34]` (~24 h @ 180 s) |
| MACRO OHLC | **7200 s** (`data_handler.granularity`) |
| MICRO (TCN) | **180 s** (`micro_granularity`) — M3 |
| Contrato | **N × 3 min** RISE_FALL (N ∈ {1,2,3,5} eleito no launch-train; placeholder **9 m** / 3 barras) |
| MINI OHLC | **180 s** (`mini_granularity`) |
| Bootstrap wait | `bootstrap_history_wait_cap_seconds` **30** (nao dorme a granularidade inteira entre retries) |
| MILI | Tick flow (nao OHLC) |
| Features | **34D** (`FEATURE_DIM`) |
| Label | `ma_trend` (MA 5; horizonte **N barras** micro = N×180 s; alinhado ao contrato promovido) |
| Online training | **false** (DEMO usa checkpoint do `launch-train`) |
| ACC / deploy | `soft_min_val_accuracy` **0.53**; `max_brier` / `soft_max_brier` **0.26**; `force_ok=false` |
| Retries | `train_deploy_retries` **5** (reseed + reset de pesos) |
| Early stop | `min_epochs` **20**, `early_stopping_patience` **16**; patience so em ganho de **val_acc** com **BCE** CE&lt;**0.70** (monitor de val ignora `focal_gamma`); restore **best val_acc** sharp |
| Meta | LightGBM **43D** `predicted_payoff_edge` |
| Sweep horizonte N | **On** no launch-train (`horizon_sweep.run_in_launch_train=true`): grade **1/2/3/5**; duration = N×3 min; elegivel **settle_wr** ≥ be+**0.03**, n≥16, history≥800; promote grava duration + `label_horizon_bars` |

## Entry points

| Comando | Papel |
|---------|-------|
| `train.py` / `app/train.py` | treino TCN |
| `app/scripts/batch/launch-train.bat` | sanitize → sweep horizonte N (H1–H5) + promote → gate → Timescale → meta |
| `app/scripts/operations/run_launch_train_tf_pipeline.py` | orquestra sweep horizonte N + promote (fallback `train.py` se `horizon_sweep.run_in_launch_train=false`) |
| `make docker-rebuild` | recarrega meta/loss apos o treino (**nao** apaga `data/dl`) |
| `app/scripts/operations/sanitize_fresh_run.py` | limpa `data/dl`, meta/loss pkls, Triton bins e estado em `data/` (so train/reset) |
| `app/scripts/operations/check_dl_deploy_gate.py` | aborta meta se geometria invalida; aceita **settle_wr** elegivel (mesmos knobs do sweep) ou ACC≥0.53 + soft path; simbolos de `settings.symbols` |
| `app/scripts/operations/train_meta_*.py` | treino offline do meta (`--source auto`; simbolos de settings) |
| `app/scripts/operations/sweep_train_timeframes.py` | loop de celulas H1–H5; artefactos em `data/dl/sweep/R_10/H{N}`; leaderboard JSON |
| `app/scripts/operations/promote_tf_winner.py` | promove vencedor elegivel para `settings.json` + `drift_symbols.py` + `data/dl` (fail-closed se nenhum) |

## Sweep de horizonte N (launch-train)

O TCN estima deslocamento em **N velas M3**. O `launch-train` **nao fixa N=3**: treina a grade `{1,2,3,5}` (contratos 3/6/9/15 min), loga `[HORIZON]` por celula e promove o mais assertivo (`settle_wr` ≥ be+0.03, n≥16, history≥800). Artefactos em `data/dl/sweep/R_10/H{N}/`. Placeholder no SSOT ate o primeiro promote: duration **9** / `label_horizon_bars` **3**.

Pipeline **offline** (nao troca N por ciclo ao vivo):

1. `horizon_sweep.n_bars` no SSOT — celulas **H1/H2/H3/H5** no relogio M3 (lookback/history copiados, sem reescalar wall-clock). Simbolo **R_10**.
2. `run_launch_train_tf_pipeline.py` limpa `data/dl/sweep`, treina cada celula com ckpt isolado (`data/dl/sweep/R_10/H{N}/`), **infra/MinIO off** no sweep, **1 tentativa** (`train_deploy_retries=1`), grava leaderboard.
3. Elegivel: **`settle_wr` ≥ be + 0.03** **e** `settle_n ≥ min_settle_n` (**16**) **e** `history_bars ≥ min_history_bars` (**800**). Label ACC e so telemetria.
4. Com `auto_promote=true` (default), promove vencedor para `settings.json` (`duration` + `label_horizon_bars`) + `data/dl/` (carimba `deploy_ok`) — **fail-closed** se nenhum elegivel (meta nao roda).
5. Gate ACC/settle + meta no SSOT promovido. Depois: `make docker-rebuild` + sync MinIO/Triton.

Knobs: `horizon_sweep.n_bars` / `run_in_launch_train` / pisos settle. Flags CLI: `--only H1 H3`, `--dry-run`, `--skip-promote`.

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

`resolve_deploy_ok` (treino) exige `val_accuracy >= soft_min` **antes** de `mini_ok` / `force_ok`. Checkpoint com ACC 0.52 grava `deploy_ok=false` no treino, mas o sweep ainda persiste settle_* para ranking. Pos-promote, `check_dl_deploy_gate` **e** o load DEMO (`_effective_deploy_ok`) aceitam o ckpt se **settle_wr** passar a elegibilidade do sweep; senao caem no path ACC/Brier/collapse. Sem isso, promote M3 com ACC&lt;0.53 gera `SKIP:DEPLOY` eterno na DEMO. Treino rejeitado **sempre sobrescreve** o `.pth` anterior (sem preservar deploy antigo). Nao usar `force_ok=true` nem `bypass_deploy_gate=true` em producao.

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

## Pos-migrate hibrido (SSOT atual M3)

1. Invalidar checkpoints `data/dl/*.pth`, TorchScript MinIO/Triton com `granularity`/`lookback` ≠ settings (ex.: legado **120**/3600 ou lookback **720**).
2. Re-hydrate Timescale (`docker-hydrate.sh` / `ensure_timescale`) para OHLC micro/MINI **180** / macro **7200**.
3. Retreinar com **`launch-train.bat`** (TCN `lookback=480`, micro **180**) + meta — **nao** via `launch-all-demo`.
4. So depois: `launch-all-demo.bat`; validar CFG live `ohlc=180s`, `macro=7200s`, `contrato=3m`.

Com `online_training=false` (SSOT), a DEMO nao agenda retreino TCN em runtime (nem settle nem rolling); usa o checkpoint do `launch-train`. Para reativar, `online_training=true` + `rolling_retrain_bars` / `retrain_min_bars` (sem `mark_force_retrain` no settle). Meta e loss-clf fazem `/v1/learn` a cada trade.

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
