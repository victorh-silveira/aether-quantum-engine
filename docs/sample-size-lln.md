# Significância Estatística e Lei dos Grandes Números (LLN)

## 1. O Vício das Pequenas Amostras

Inspirado nas pesquisas de Tversky, Kahneman e Mlodinow, o Aether reconhece que amostras reduzidas (menos de 30 eventos) são dominadas pela variância aleatória. Streaks de vitória ou derrota em curto prazo não representam vantagem ou perda de edge.

---

## 2. Critérios de Amostragem do Aether

Para proteger o capital e evitar conclusões precipitadas:

1. **Tamanho Mínimo de Promoção:**
   - Para que uma versão de setup seja promovida de `piloto` para `ativo` na conta REAL, são exigidos pelo menos **200 contratos DEMO liquidados pelo broker**.
2. **Intervalo de Confiança:**
   - O retorno médio líquido obtido nos 200 contratos deve apresentar o limite inferior unilateral de 95% estritamente positivo:
     $$IC_{95\%}(\mu) = ar{x} - 1,645 \cdot rac{s}{\sqrt{n}} > 0$$
3. **Liquidações Reais vs. Backtest:**
   - Apenas contratos efetivamente submetidos e liquidados pela corretora qualificam a amostra. Backtests com proxies de velas são apenas indicativos analíticos.
