Para entender como Jim Simons e a equipe da Renaissance Technologies (RenTec) abordariam esse cenário específico por meio do fundo **Medallion**, é preciso primeiro desmistificar a narrativa tradicional de Wall Street. Jim Simons nunca operaria com base em narrativas macroeconômicas discricionárias do tipo _"o EURUSD subiu, então vou comprar ações porque é Risk-On"_.

A RenTec aborda o mercado estritamente como um **sistema de processamento de sinais ruidosos**. O Medallion buscaria anomalias estatísticas, microestruturais e matemáticas escondidas na dinâmica de preços desse conjunto específico de ativos em um horizonte de tempo estrito de 15 minutos.

Abaixo está o detalhamento metodológico e matemático de como o Medallion estruturaria esse modelo preditivo cross-asset (inter-mercados) utilizando o par frxEURUSD como a variável preditora primária para os índices americanos (OTC\_SPC, OTC\_NDX, OTC\_DJI) e europeus (OTC\_FCHI, OTC\_GDAXI, OTC\_SSMI, OTC\_FTSE).

1\. A Lógica Quantitativa: O EURUSD como Proxy de Liquidez Global
-----------------------------------------------------------------

No framework quantitativo, o par frxEURUSD não é apenas uma taxa de câmbio; ele funciona como o principal termômetro de liquidez global e diferencial de taxa de juros de curto prazo entre o Federal Reserve (Fed) e o Banco Central Europeu (BCE).

*   **Regime de Risk-On (Apetite ao Risco):** Tradicionalmente correlacionado com o enfraquecimento do Dólar Americano ($USD$). Em momentos de otimismo, o capital sai de ativos de refúgio (como os títulos do Tesouro dos EUA e o próprio dólar) e migra para ativos globais. Isso gera um fluxo comprador no Euro, empurrando o frxEURUSD para cima.
    
*   **Regime de Risk-Off (Aversão ao Risco):** Ocorre a repatriação de capital para o $USD$ ("flight to safety"). O Euro se desvaloriza frente ao dólar, derrubando o frxEURUSD.
    

O Medallion não operaria essa correlação de forma linear ou estática. O fundo exploraria o **descompasso temporal** (lead-lag effect) em barras de altíssima frequência e horizontes de 15 minutos.

2\. Modelagem Matemática e Arquitetura do Sinal
-----------------------------------------------

Para prever a direção dos próximos 15 minutos, o Medallion decomporia as séries temporais dos ativos usando técnicas avançadas de processamento de sinais e aprendizado estatístico.

### Cadeias Ocultas de Markov (Hidden Markov Models - HMM)

Na literatura RenTec, HMM com Baum-Welch identifica estados ocultos que alteram a dinâmica dos retornos. No horizonte de 15 minutos do Aether, **duas camadas** coexistem sem se confundir:

| Camada | O que modela | Origem no motor |
|--------|----------------|-----------------|
| **Macro transatlântico** | Risk-On / Risk-Off / divergência US-EU | Voto quantitativo dos índices US e EU em M15 (`classify_transatlantic_confluence`) |
| **HMM no marcapasso** | Volatilidade e persistência de regime no `frxEURUSD` | `MarketHMMClassifier`: estado 0 = reversão à média, estado 1 = tendência/rompimento |

O EURUSD permanece variável observável central; o HMM **não** substitui as tags `US_CLUSTER` / `EU_CLUSTER` da LLM. Ele modula convicção via StatArb (`llm_macro_confluence_guards`). Detalhes em [`arquitetura.md`](arquitetura.md).

### Arbitragem Estatística Cross-Asset e Cointegração

Em vez de olhar apenas para o preço bruto, o Medallion calcularia os resíduos de modelos de regressão dinâmicos. Embora moedas e índices de ações não sejam perfeitamente cointegrados no longo prazo devido a fatores estruturais diferentes, eles compartilham componentes estocásticos comuns no curto prazo (janelas móveis de volatilidade).

O modelo estimaria uma função de transição onde o retorno esperado de um índice europeu (ex: OTC\_GDAXI) ou americano (ex: OTC\_SPC) no período $t + 15\\text{min}$ é uma função dos desvios da média móvel do frxEURUSD e do próprio índice:

$$\\Delta \\text{Índice}\_{t \\to t+15} = \\alpha + \\sum\_{i=0}^{k} \\beta\_i \\Delta \\text{EURUSD}\_{t-i} + \\gamma (\\text{Índice}\_t - \\theta \\text{EURUSD}\_t) + \\epsilon\_t$$

Onde:

*   $\\theta$ é o coeficiente de hedge dinâmico.
    
*   $\\gamma$ é a velocidade de reversão à média do desequilíbrio gerado pelo fluxo de ordens.

**Implementação no Aether:** a equação acima é aproximada por `compute_pca_cointegration_zscores` (fator comum PC1 + resíduo por índice em janela M15). A seleção do índice no cluster ativo usa o Z-Score alinhado à tag LLM (`cluster_statarb_select.py`).

3\. Matriz de Correlação Dinâmica e Execução nos Índices
--------------------------------------------------------

Como o Medallion distribuiria as ordens entre os ativos com base nas variações do frxEURUSD em 15 minutos? O fundo aproveitaria assimetrias geográficas e de sensibilidade ao risco ("Beta").

| Ativo | Região | `frxEURUSD` ↑ (Risk-On) | `frxEURUSD` ↓ (Risk-Off) |
|-------|--------|-------------------------|---------------------------|
| `OTC_SPC` (S&P 500) | US | Alta moderada (beta de mercado) | Queda sistemática |
| `OTC_NDX` (Nasdaq 100) | US | Alta agressiva (juros/liquidez) | Queda acentuada |
| `OTC_DJI` (Dow Jones) | US | Alta defensiva (valor/industriais) | Queda moderada |
| `OTC_GDAXI` (DAX) | EU | Alta forte (exportador) | Queda severa |
| `OTC_FCHI` (CAC 40) | EU | Alta consumo/luxo | Queda correlacionada Europa |
| `OTC_SSMI` (SMI) | EU | Alta limitada (CHF refúgio) | Desempenho relativo defensivo |
| `OTC_FTSE` (FTSE 100) | EU | Misto (commodities em USD) | Reação mista (mineradoras/energia) |

Execução no motor: `risk_on` → cluster US; `risk_off` → cluster EU; um índice por cluster via StatArb quando habilitado (`docs/arquitetura.md` §6).

### Assertividade e drawdown (motor Aether)

O Medallion no Aether prioriza **menos trades de maior qualidade** em vez de volume cego:

- **Macro:** piso de força em `risk_on`/`risk_off`; divergência só com líder forte; `indefinido` bloqueado sem gap US/EU; `allowed_execute_tags` limita regimes executáveis.
- **Simbolos:** `excluded_symbols` remove indices fracos do cluster (ex.: SPC, FTSE, NDX).
- **LLM live:** `llm.refresh_schedule=tag_change` consulta Gemini quando a tag macro muda; `llm.refresh_interval_hours` força reconsulta periódica (ex.: 4h).
- **Sessão OTC:** `trading.session` limita entradas à janela UTC, warm-up após sync de velas e bloqueio nos minutos finais antes do fechamento.
- **StatArb:** Z contra a direção em HMM de reversão → veto (`STATARB_VETO`); alinhamento → boost de convicção; até 2 índices por cluster (`statarb_index_max_per_cluster`).
- **Risco:** Kelly com `max_stake_pct` / `max_stake_pct_high_conviction`, cooldown menor em convicção alta e **freio de drawdown** por sessão; stop win diário 10% (conta grande) ou valor fixo (conta pequena).

Parâmetros em `config/settings.json` → `strategy.macro` e `risk_management.kelly`. Backtest e live compartilham os mesmos guardrails (`llm_macro_confluence_guards.py`).