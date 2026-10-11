# Modelos e limites dos sinais

Cada par ativo/período usa oito retornos logarítmicos causais das últimas nove velas fechadas. O treino offline separa cronologicamente 70% das amostras para ajuste e 30% para validação. Exige pelo menos 200 amostras. Os parâmetros de normalização vêm somente do trecho de treino. O modelo é **APROVADO** quando supera a probabilidade de maioria do treino em acurácia e Brier; caso contrário, é **EXPERIMENTAL**. Este último não aparece como sinal ativo nos painéis; o catálogo mostra `SEM SINAL` e o estado experimental.

O treinamento busca até 12 páginas de velas M5 públicas (cerca de 12 mil por índice) somente dos índices Volatility disponíveis (`R_<número>` e `1HZ<número>V`) e agrega M15/H1 sem completar lacunas. O relatório mostra amostras de validação, acurácia e Brier do modelo e da referência com cinco casas decimais, além dos totais por estado. Também registra o horário UTC da última vela fechada de cada período. Repetir o treino antes de novas velas fecharem pode produzir os mesmos resultados. Um checkpoint aprovado anterior é arquivado em `data/indicator-models/rejected/` quando nova avaliação o reclassifica como experimental ou sem modelo. Falhas transitórias de coleta não fazem essa revisão.

O treino sincroniza as versões ativas no Timescale ao concluir. O indicador compara a próxima vela com o fechamento da vela anterior; não prevê o resultado de contratos Rise/Fall abertos a qualquer horário, pois esses contratos comparam o preço de entrada com o preço no vencimento.

Artefatos incompatíveis não são carregados. A classificação experimental não demonstra vantagem futura nem rentabilidade. A coluna `correct` compara a direção prevista com o próximo fechamento do mesmo sintético. Não representa liquidação de contrato ou previsão de Forex.

`APROVADO` descreve somente a comparação na janela histórica de validação; cada previsão futura pode falhar. O painel mostra separadamente o número de resultados já confirmados da versão atual. Durante a vela prevista, o preço pode estar `A FAVOR` ou `CONTRA` em observações provisórias coletadas a cada 30 segundos; o resultado só é fixado no fechamento do período.
