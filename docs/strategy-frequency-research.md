# Pesquisa de frequência M15 com validação DEMO

A pesquisa compara oportunidades nos 12 ativos OTC sem alterar a configuração operacional de `persistent_trend_v3`. Seu objetivo é maximizar a média diária de ordens selecionadas com evidência de EV positivo, aceitando menor vantagem por operação. Resultados por OHLC são proxies: não constituem contratos liquidados pela Deriv nem qualificam operação REAL.

## Reprodução

Execute os comandos no WSL, a partir da raiz do repositório, usando o ambiente Conda `deriv-api`:

```bash
PY=/mnt/c/Users/victor-silveira/anaconda3/envs/deriv-api/python.exe
"$PY" -m scripts.analysis.frequency_data --output .cache/frequency-20261125
"$PY" -m scripts.analysis.frequency_search --output .cache/frequency-20261125 --stage all
```

O coletor congela a última fronteira M15 fechada em `cutoff_epoch.txt`. Reexecuções reutilizam o mesmo corte; `--cutoff EPOCH` permite declarar explicitamente outro instante. Cada par ativo/período possui arquivo Parquet, inventário e SHA-256; `candles.parquet` e `manifest.json` identificam o conjunto consolidado. Duplicatas conflitantes, OHLC inválidos, velas abertas e lacunas são inventariados. Velas de sessões fechadas não são imputadas. O aquecimento antecede 01/01/2026.

O diretório de saída também contém a identidade da execução, a grade, os resultados agregados e replays detalhados. Ele é um artefato local de pesquisa, não uma configuração operacional. Os comandos acima exigem acesso à API pública e não usam credenciais de negociação.

## Hipóteses e avaliação causal

- A grade nominal possui 220 combinações: cinco ramos individuais, união dos cinco e cinco ablações, multiplicadores DI `0,50 / 0,625 / 0,75 / 0,875 / 1,00` e ADX mínimo `20 / 25 / 30 / 35`.
- ADX não interfere em composições sem B5. Essas duplicatas são removidas antes dos resultados, restando 145 hipóteses distintas, identificadas por parâmetros imutáveis e hash. A composição completa com escala DI `1,00` e ADX `35` reproduz a v3.
- Os comparadores fixos `dual_sniper`, `multi_regime_hf` e `senior_institutional` acrescentam três hipóteses, totalizando 148. Não são criadas novas combinações após observar os resultados.
- DI/ADX preservam a reinicialização da janela de até 500 velas usada em produção. H1 usa exclusivamente horas encerradas no instante da entrada. Os sinais vetorizados são confrontados com o avaliador escalar.
- Entradas e resultados exigem M15 contíguo, aquecimento e vela de resultado totalmente encerrada até o corte. Mantêm-se veto de conflitos, horas tóxicas e calendário econômico.

O calendário é aplicado pelo mesmo `evaluate_calendar_veto` de produção. O feed FairEconomy/ForexFactory atual é semanal. Seu cache não comprova cobertura de janeiro até novembro; períodos sem arquivo histórico verificável permanecem não qualificados. Ausência de eventos históricos não é interpretada como prova de ausência de eventos nem produz vetos artificiais.

## Replay e inferência

O replay seleciona no máximo quatro ordens por ciclo. Reutiliza `available_cluster_bankroll` e `pullback_stake`, com stake base de 1,5%, orçamento de 4%, mínimo absoluto de uma unidade e recuperação desativada. Compras reservam saldo sequencialmente; ganhos e perdas do mesmo ciclo são liquidados em conjunto antes do próximo ciclo. São comparados payouts hipotéticos de 70%, 72,35% e 80%, com bancas iniciais de 1.000 e 7.000 unidades monetárias.

O mínimo absoluto pode exceder o orçamento percentual em bancas pequenas. O replay conta essas ocorrências em `cluster_budget_breaches`, sem ocultar a diferença entre o limite declarado e o comportamento do sizing existente. Uma ocorrência impede a seleção do candidato.

Sem snapshots históricos de propostas e evidência broker, o ranking usa o proxy determinístico FTSE primeiro e desempate lexical. A produção também considera evidência, probabilidade, cotação, liquidez e estado de sessão; empates podem depender da conclusão assíncrona das propostas. Portanto, ordens selecionadas no proxy não são apresentadas como compras garantidas no runtime. `Signal.selection_score` permite fornecer scores observados, quando disponíveis.

A janela inicial de pesquisa é janeiro–março. O walk-forward é expansivo, com seleção sobre o passado e avaliação em cada mês de abril a outubro. Novembro é diagnóstico recente separado. Tanto 2026 quanto novembro estão incompletos; além disso, 2026 já influenciou a seleção da v3 e não constitui histórico inteiramente intocado.

A elegibilidade estatística requer no mínimo 200 operações fora da amostra, distribuídas em 30 dias ativos, EV com limite inferior unilateral de 95% positivo no payout de 70% e teste ajustado por Holm ao nível de 5% para a família completa. O bootstrap centrado usa blocos circulares de cinco dias, preservando conjuntamente os ativos e dias sem trades, 10.000 réplicas e semente 42. A frequência é calculada sobre todos os dias da janela, inclusive dias zerados.

Entre candidatos aprovados, o critério é maior média diária, seguido de maior limite inferior do EV e menor drawdown. Drawdown e sequências de perdas são comparativos, sem introduzir stop financeiro novo. O drawdown refere-se ao patrimônio liquidado; a sequência intracluster depende do desempate determinístico.

## Limites de promoção e testes

`report.json` registra as limitações e a seleção ou sua ausência. `demo_candidate.json` é um descritor de pesquisa, não um arquivo que habilita negociações. Sem dados plenamente qualificados e aprovação estatística, nenhum vencedor é promovido. Uma eventual variante deve ter identidade própria, preservando v3 e rollback.

Um candidato futuro exige pelo menos 200 contratos DEMO prospectivos liquidados pelo broker. Nenhum trade OHLC substitui essa amostra. Os gates de cotação e evidência para REAL permanecem vigentes.

```bash
"$PY" -m pytest tests/analysis --basetemp=.cache/pytest-frequency
"$PY" -m ruff check scripts/analysis tests/analysis
"$PY" -m ruff format --check scripts/analysis tests/analysis
```

Os testes verificam paridade vetorial/escalar de todas as 145 hipóteses, fronteiras DI/ADX, aquecimento, lacunas, H1 causal, calendário, ordem de seleção, orçamento, reservas e liquidações simultâneas. A execução completa também deve satisfazer os gates de qualidade do projeto e cobertura de 100% em `src`.

## Resultado da execucao de 25/11/2026

Coleta de 198.849 velas M15/H1 dos 12 OTC, congelada em 25/11/2026 as 17:00 BRT. SHA-256 do Parquet consolidado: bbc4dadbf56ff250665ec8e23a9200d3142aff23ae687ce8337af89cbd8a0d1c. Artefatos completos em .cache/frequency-20261125/.

Nenhuma das 148 hipoteses satisfez os criterios estatisticos. O walk-forward de abril a outubro tambem nao selecionou candidato em nenhum dos sete meses usando apenas o passado. A cobertura anual do calendario nao foi comprovada; o aquecimento H1 anterior a janeiro variou de 159 a 474 velas, limitado pelo historico publico. A configuracao operacional permanece na v3 e demo_candidate.json fica desabilitado.

Comparacao fora da amostra retrospectiva, abril a outubro, 214 dias, payout hipotetico de 70%, banca inicial de 1.000:

- V3 contigua: 343 ordens; 1,60 por dia; EV +0,0178u por ordem; limite inferior de 95% -0,0482u; drawdown 18,30%.
- Maior frequencia, b12345_di0.5_adx30: 2.513 ordens; 11,74 por dia; EV -0,1005u; limite inferior -0,1281u; drawdown 98,14%. O replay tambem detectou violacoes do budget pelo minimo absoluto apos forte reducao da banca; hipotese rejeitada.
- dual_sniper: 1.683 ordens, 7,86 por dia, EV -0,0857u e drawdown de 90,18%.
- multi_regime_hf: 1.977 ordens, 9,24 por dia, EV -0,1273u e drawdown de 98,08%, com violacoes do budget em banca reduzida.
- senior_institutional: nenhum sinal elegivel nessa janela.

Dias zerados incluem tanto ausencia de sinal quanto impossibilidade de executar por saldo/stake. As contagens brutas e os motivos de rejeicao ficam separados nos replays JSON; a media acima nao e promessa de ritmo operacional.

### Reconciliacao do relatorio fornecido

Hoje, ate o corte original de 16:00 BRT e ate o novo corte de 17:00 BRT, a v3 gerou zero trades. Em novembro, a regra estrita de contiguidade resulta em 59 trades, 38 wins e 21 losses, com +6,4930u a 72,35%.

O avaliador legado com allow_session_gaps=True aceita ainda um CALL vencedor de OTC_GDAXI em 12/11 as 04:15 BRT (epoch 1794467700), apos lacuna anterior de 38.700 segundos. A vela de resultado e contigua, de 26861,5 para 26878,0. Incluindo essa entrada, os totais se reconciliam com os 60 trades, 39 wins e 21 losses informados (+7,2165u). A pesquisa exclui essa entrada para cumprir o mandato de M15 sem gaps; a tolerancia legada de producao nao foi alterada nesta mudanca.

No ano parcial, a v3 contigua totalizou 561 trades (335 wins, 221 losses e 5 empates OHLC). Empates valem zero no proxy; a regra de liquidacao de um contrato broker nao e inferida dessa convencao.
