# Lei dos Grandes Numeros no Aether

Politica operacional contra o vies dos pequenos numeros (Mlodinow / Tversky-Kahneman).

## SSOT

`config/settings.json` → `orchestrator.execution.sample_size_policy`

| Knob | Padrao | Efeito |
|------|--------|--------|
| `evidence_n_min` | 20 | Minimo para misturar live_wr no Kelly e declarar underperformance |
| `large_n_min` | 40 | Amostra grande (soft SIDE_EQ / ranking) |
| `n_min_small` (side_eq) | 8 | Abaixo disso: `small_n_insufficient` (pass) |
| `calib_soft_min_n` | 15 | Abaixo: sem `CALIB_DRIFT_SOFT` |
| `toxic_side_n_min` | 8 | Toxic escape / flip so com N do lado |
| `explore_stake_scale_floor` | 0.40 | Teto relativo de stake EXPLORE com N=0 |
| `z_sig_threshold` | 1.64 | Evidencia binomial (~90%) para hard-skip por WR |

## Fluxo

1. Cold start (`live_n` baixo): stake EXPLORE reduzida; prior de conviccao domina Kelly.
2. SIDE_EQ: nao trata 2 losses como prova; exige janela e, para WR puro, z significativo.
3. Calib drift: ECE alto com N=1 e ignorado ate `calib_soft_min_n`.
4. Large-N: soft penalty Kelly/margem, sem flip automatico por “mao quente”.

Ver tambem [`medallion.md`](medallion.md) secao 1.1.
