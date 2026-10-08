# Modelos e limites dos sinais

Cada par ativo/período usa oito retornos logarítmicos causais das últimas nove velas fechadas. O treino offline separa cronologicamente 70% das amostras para ajuste e 30% para validação. A regressão logística só é publicada quando supera a probabilidade de maioria do treino **em acurácia e Brier** na validação. Exige pelo menos 200 amostras. Os parâmetros de normalização vêm somente do trecho de treino.

O treinamento usa velas M5 públicas paginadas e agrega M15/H1 sem completar lacunas. Modelos rejeitados ou incompatíveis não são carregados. Isso reduz sinais sem evidência mínima, mas não demonstra vantagem futura nem rentabilidade. A coluna `correct` compara a direção prevista com o próximo fechamento do mesmo sintético. Não representa liquidação de contrato ou previsão de Forex.
