# Touch/No Touch: dados, treino e operacao

O SSOT `config/settings.json`, secao `touch`, ativa contratos ONETOUCH e
NOTOUCH em 1HZ75V com duracao de 300 segundos. DEMO e REAL usam a mesma
logica; somente autenticacao/conta mudam. Nao existe fallback para CALL/PUT.

## Modelo e qualificacao

O primeiro modelo dedicado e uma regressao logistica padronizada com
calibracao isotonica em janela separada, exportada em JSON. Nao e o TCN nem
o meta direcional renomeado. As cinco features causais representam distancia
da barreira, drift, volatilidade, razao de volatilidade e frequencia de ticks.
Resultados Touch nao alimentam os learners direcionais CALL/PUT.

O alvo e o primeiro toque inclusivo na barreira durante toda a trajetoria,
nao o fechamento M5. O replay usa ticks observados e latencias de 0/1/2s;
lacunas superiores a 1,5s invalidam a janela. Para cada decisao sao cotadas
ambas as familias nas mesmas barreiras absolutas. EV usa ask/payout da
proposta, sem payout fixo e sem inverter uma probabilidade direcional.

Treino/calibracao/teste sao cronologicos (60/20/20), agrupados por decisao
e separados por purging da janela historica, duracao e latencia. Exigem-se
pelo menos 600 grupos; isso nao garante 120 trades OOS selecionados.
Qualificacao exige pelo menos 120 trades OOS nao sobrepostos, limite inferior
unilateral de 90% do retorno positivo e Brier melhor que o baseline constante.
O retorno avaliado e o pior entre as latencias testadas, usando payout cotado.

`quoted_tick_replay` e simulacao com cotacoes reais: nao equivale a contrato
executado auditado, nao prova lucratividade futura e nao modela toda a cauda
da latencia. O intervalo t pressupoe aproximacao de independencia; regimes,
dependencia residual e repeticao de experimentos podem tornar a estimativa
otimista. Nao reutilize o mesmo holdout para escolher varias estrategias.

## Preparacao e coleta sem ordens

1. Aplique `infra/docker/008_touch_contract_audit.sql` no Timescale existente
   antes de iniciar o motor. A inicializacao de volumes novos e o lifecycle
   incluem essa migracao idempotente; nao apague volumes para aplica-la.
2. Na raiz do repositorio, com o ambiente `deriv-api` e Timescale disponivel,
   execute `python app/scripts/operations/collect_touch_dataset.py`.
   Esse coletor publico assina ticks e solicita propostas; nao compra contratos.
3. Aguarde dados suficientes e rode o `launch-train` habitual. O launcher
   desvia para `train_touch_classifier.py` antes do sanitize/TCN/meta legado.

Ticks ficam no Timescale; propostas em `data/touch/quotes.jsonl`. O sanitize
preserva `data/touch`. `--max-groups N` limita a coleta e aguarda a janela
final de settlement; Ctrl+C encerra, podendo deixar a ultima janela incompleta.
Queda de WSS encerra a coleta explicitamente; reinicie e nao interpole lacunas.

Sem propostas/ticks suficientes, o treino termina com erro informativo.
Exportar `data/touch/model.json` pode concluir com `qualified=false`: isso
significa experimento concluido, nao autorizacao para compras. Sao rejeitados
artefatos ausentes, vencidos (7 dias), incompativeis ou nao qualificados.

## Execucao

O motor coleta propostas mesmo sem modelo aprovado. Com modelo aprovado,
seleciona EV acima de 0,03, verifica frescor de ate 2s e compra exatamente o
ID cotado via WSS. Stake limitada ao menor teto entre 0,1% da banca, limite
absoluto configurado e Kelly fracionario; nao eleva stake ao minimo se este
exceder o teto. O filtro Kelly adicional pode reduzir os trades em relacao
ao replay OOS. Nao ha recovery sizing ou compra REST inferida pelo saldo.

Falha apos envio de buy mantem trava de confirmacao incerta durante a sessao;
reconcilie os contratos antes de reiniciar o motor. Nao ha retry de compra.
Auditoria persiste ID da proposta, tipo, barreira e campos efetivamente
retornados pelo broker. Campos ausentes nao sao inventados.

A migracao de software nao fornece historico nem vantagem estatistica.
Nenhum comando de coleta ou treino precisa abrir ordens para produzir dados.
