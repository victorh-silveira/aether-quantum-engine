---
name: aether-single-strike-risk
description: >-
  Audita e calibra o dimensionamento de risco Kelly Single-Strike projetado para atingir a meta
  de Stop-Win de 1% da banca em uma unica tacada M15 com payout real de 85%.
---

# Kelly Single-Strike Risk Management (M15 / Stop-Win 1%)

Especialista em dimensionamento de lote e gestão matemática da esteira de risco para bater 1% da banca em um único trade M15.

## Formulação Matemática
$$\text{Lucro Alvo} = \text{Banca} \times 0.01$$
$$\text{Stake Single-Strike} = \frac{\text{Lucro Alvo}}{\text{Payout}} = \frac{\text{Banca} \times 0.01}{0.85} \approx 0.011765 \times \text{Banca} \approx 1.18\% \text{ da banca}$$

## Parâmetros SSOT (`config/settings.json`)
- `risk_management.params.compounding_rate_daily = 0.01` (1% ao dia)
- `risk_management.params.payout_estimate = 0.85`
- `risk_management.kelly.default_payout = 0.85`
- `risk_management.kelly.stop_win_kelly_cycles_target = 1` (tacada única)
- `risk_management.kelly.stop_win_kelly_min_fraction = 1.0`
- `risk_management.kelly.stop_win_kelly_max_fraction = 1.0`
- `risk_management.kelly.max_stake_pct = 0.05` (permite a stake de 1.18% com teto de segurança)

## Regras Operacionais
1. Quando a convicção atende ao piso de entrada (`stop_win_kelly_min_conviction = 0.52`), a stake é automaticamente dimensionada para atingir o restante da meta de 1%.
2. Atingida a meta de 1%, o stop-win dispara imediatamente com status `STOP_WIN` e o bot encerra as operações da sessão.
3. Não há progressão descontrolada ou revenge sizing pós-loss.
