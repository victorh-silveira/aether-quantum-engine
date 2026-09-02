---
name: aether-single-strike-risk
description: >-
  Audita e calibra o dimensionamento de risco Kelly Single-Strike projetado para atingir a meta
  de Stop-Win de 4.31% da banca em uma unica tacada M5 com payout real de 85%.
---

# Kelly Single-Strike Risk Management (M5 / Stop-Win 4.31%)

Especialista em dimensionamento de lote e gestão matemática da esteira de risco para bater 4.31% da banca em um único trade M5 (equivalente a 3% ao dia em 21 dias úteis).

## Formulação Matemática
$$\text{Lucro Alvo} = \text{Banca} \times 0.0431$$
$$\text{Stake Single-Strike} = \frac{\text{Lucro Alvo}}{\text{Payout}} = \frac{\text{Banca} \times 0.0431}{0.85} \approx 0.0507 \times \text{Banca} \implies \text{Cap em } 5.0\% \text{ da banca}$$

## Parâmetros SSOT (`config/settings.json`)
- `risk_management.params.compounding_rate_daily = 0.0431` (4.31% por sessão)
- `risk_management.params.payout_estimate = 0.85`
- `risk_management.kelly.default_payout = 0.85`
- `risk_management.kelly.stop_win_kelly_cycles_target = 1` (tacada única)
- `risk_management.kelly.stop_win_kelly_min_fraction = 1.0`
- `risk_management.kelly.stop_win_kelly_max_fraction = 1.0`
- `risk_management.kelly.max_stake_pct = 0.05` (teto seguro de 5.0% da banca)

## Regras Operacionais
1. Quando a convicção atende ao piso de entrada (`stop_win_kelly_min_conviction = 0.52`), a stake é automaticamente dimensionada para atingir a meta de 4.31%.
2. Atingida a meta de 4.31%, o stop-win dispara imediatamente com status `STOP_WIN` e o bot encerra as operações da sessão (`EXEC_PAUSE`).
3. Não há progressão descontrolada ou revenge sizing pós-loss. Soft recovery suave em 2 a 3 ciclos (`cover_multiple: 1.10`, `max_safe_stake_pct: 0.035`).

