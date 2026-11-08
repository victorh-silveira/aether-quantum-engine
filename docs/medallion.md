# Metodologia Quantitativa e Gestão de Capital

## 1. Princípios Medallion Aplicados ao Aether

O Aether Quantum Engine adota a postura quantitativa rigorosa inspirada na gestão sistemática de fundos quantitativos:

1. **Sinais Determinísticos, Não Narrativas:** O mercado é tratado como uma série temporal com ruído estocástico. Não operamos com base em expectativas macroeconômicas discricionárias, mas sob padrões estatísticos repetíveis.
2. **Gestão de Risco Convexa:** A sobrevivência a longo prazo depende de limitar severamente o drawdown máximo, permitindo que a assimetria estatística se expresse no longo prazo.
3. **Veto Financeiro em Nível de Tick:** A vantagem teórica de um setup técnico é inútil se o broker cotar um payout degradado. A entrada só se justifica se houver margem econômica comprovada sobre o payout líquido.

---

## 2. A Matemática do Valor Esperado (EV)

Em contratos de opções binárias com retorno fixo de payout $R$ e probabilidade de acerto $p$:

$$	ext{Break-Even Rate} = rac{1}{1 + R}$$

$$	ext{Valor Esperado (EV)} = p \cdot (1 + R) - 1$$

- Para autorização de contratos em conta REAL, o sistema exige:
  $$p \cdot (1 + R) - 1 \ge 0,01$$
- Se a corretora ofertar payout abaixo do limiar que assegura $EV \ge 0,01$, a ordem é cancelada instantaneamente no gate de proposta.

---

## 3. Dimensionamento de Lote e Conservação de Capital

- **Fração Base:** $f^* = 1\%$ da banca líquida disponível.
- **Amortização Linear de Perdas:**
  Em caso de prejuízo acumulado $L$, a recomposição busca retornar ao ponto neutro sem alavancagem excessiva:
  $$	ext{Stake} = \min\left(rac{L}{R}, 5\% 	imes 	ext{Banca}ight)$$
- O teto de 5% impede que sequências adversas degradem a banca em ritmo superior à capacidade de absorção matemática de variância.
