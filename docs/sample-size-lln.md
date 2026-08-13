# Lei dos Grandes Numeros no Aether

Politica operacional contra o vies dos pequenos numeros (Mlodinow / Tversky-Kahneman).

## SSOT

`config/settings.json` → `orchestrator.execution.sample_size_policy`

| Knob | Padrao | Efeito |
|------|--------|--------|
| `evidence_n_min` | 12 | Minimo para misturar live_wr no Kelly e declarar underperformance |
| `large_n_min` | 32 | Amostra grande (soft SIDE_EQ / ranking) |
| `n_min_small` (side_eq) | 4 | Abaixo disso: `small_n_insufficient` (pass) |
| `calib_soft_min_n` | 12 | Abaixo: sem `CALIB_DRIFT_SOFT` |
| `toxic_side_n_min` | 4 | Toxic escape / flip so com N do lado |
| `explore_stake_scale_floor` | 0.40 | Teto relativo de stake EXPLORE com N=0 |
| `z_sig_threshold` | 1.64 | Evidencia binomial (~90%) para hard-skip por WR |

## Fluxo

1. Cold start (`live_n` baixo): stake EXPLORE reduzida; prior de conviccao domina Kelly.
2. SIDE_EQ: nao trata 2 losses como prova; exige janela e, para WR puro, z significativo.
3. Calib drift: ECE alto com N=1 e ignorado ate `calib_soft_min_n`.
4. Large-N: soft penalty Kelly/margem, sem flip automatico por “mao quente”.

## SIDE_EQ runtime = soft Kelly only

No live, `hard_skip` de dominio em `side_equilibrium` e mapeado para **soft `kelly_mult`** em `execution_side_eq_sizing.apply_side_eq_kelly_sizing` (finalize do direction resolver). Nao ha SKIP/veto de direcao por SIDE_EQ; `side_eq_blocked` nao zera stake. Config: `orchestrator.execution.side_equilibrium.enabled=true`.

Ver tambem [`medallion.md`](medallion.md) secao 1.1 / 8.5.
