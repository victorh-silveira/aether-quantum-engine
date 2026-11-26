# Treinamento temporal Laya para R_10 M5

O fluxo offline usa Laya classico 0.5.5, `vegaml=False`, sem executor de ordens. Preparar os dados e validar tokens funciona em CPU; treinar exige CUDA com pelo menos 3 GiB livres. O perfil `auto` usa encoder congelado (`head`) em GPUs de ate 12 GiB, incluindo a RTX 4060 de 8 GiB; em GPUs maiores usa `full`, que exige ao menos 12 GiB livres. Nao existe fallback de treinamento pesado em CPU.

## Comandos no WSL com Conda deriv-api

```bash
make app-cuda-install
make app-cuda-check
make app-cuda-train-check
make app-prepare
make app-train-dry-run
make app-train
make app-evaluate
make app-promote
make app-run
```

O Makefile resolve o Python do Conda `deriv-api` e usa `LAYA_DATA=.cache/laya-r10-gpu-20261126`, `LAYA_RUN=.cache/laya-r10-gpu-run` e `LAYA_TRAIN_PROFILE=auto` por padrao. Alterar diretorios com `make app-train LAYA_DATA="caminho dos dados" LAYA_RUN="nova execucao"`. Preparacao de dados existentes nao deve ser repetida: o manifesto ja fixa o corte. `make app-evaluate LAYA_DEVICE=cpu` permite avaliacao explicitamente em CPU. `LAYA_ARGS` acrescenta argumentos de uma etapa, como `--cutoff` no prepare.

`app-cuda-check` prova inferencia real no device CUDA; `app-cuda-train-check` prova backward e atualizacao de parametros em memoria, sem gravar pesos ou ordens. `app-run` verifica CUDA antes de executar run.py, que continua usando o checkpoint base ou o caminho aprovado configurado em settings. Promover nao ativa automaticamente um checkpoint.

Copiar dados e checkpoint base para outro host GPU preserva os manifestos. `make app-train LAYA_TRAIN_PROFILE=full` solicita treinamento completo e exige a margem de memoria correspondente.

## Dados e ausencia de vazamento

`prepare_request.json` fixa o corte na primeira tentativa. O coletor reutiliza parquet local R_10/300s e consulta paginas publicas da Deriv, sem PAT. Precos sinteticos ou interpolados sao proibidos: lacunas confirmadas pela API excluem exemplos cuja janela de 120 barras as atravesse. Historico anterior esgotado interrompe a preparacao.

Sao 90 dias mais 120 barras de aquecimento. A divisao por tempo e 60% treino, 10% calibracao, 10% selecao e 20% teste; horizontes que alcancem a proxima divisao sao purgados. Empates de fechamento sao contados e excluidos do alvo binario. O contexto inclui 12 velas, o mesmo serializador e os mesmos indicadores causais do runtime. Os hashes vinculam perguntas, configuracoes sanitizadas e dados; mudancas exigem novo dataset.

O dry-run verifica todos os tokens, sem pesos. O orcamento inclui pergunta e opcoes e nao pode ultrapassar a capacidade do encoder. Os dados de teste sao inspecionados apenas para integridade e comprimento nessa etapa.

## Treinamento GPU e selecao

Treinar seeds 17, 42 e 73 com RLCD, quatro epocas, microbatch 1, acumulacao 64, encoder LR 0,000025 e head LR 0,0001; usar precisao mista e gradient checkpointing. No perfil head, o encoder fica congelado e somente a cabeca aprende; o manifesto registra o perfil efetivo. Cada seed salva `checkpoint_latest` ao final de cada epoca para preservar pesos de diagnostico em caso de falha. O comando recusa sobrescrever uma execucao anterior; iniciar outra exige diretorio novo.

A calibracao usa exclusivamente a faixa temporal reservada, sem sorteio do holdout padrao da biblioteca. Selecionar maior acuracia na faixa de selecao, desempatar pelo menor Brier e depois pela ordem das seeds. Salvar pesos, tokenizer, perguntas, temperaturas, identidade da base e manifesto de treinamento.

## Teste final, promocao e limites

`evaluate` exige candidato previamente selecionado e consome o teste uma unica vez, marcado por `test_consumed.json`. Nao treinar novos candidatos nesse dataset depois disso. Uma falha de avaliacao conserva o marcador e impede ajuste oportunista; investigar e preparar um novo dataset independente.

Aprovacao exige simultaneamente: melhora de pelo menos 1 ponto percentual de acuracia sobre a base, acuracia superior a classe majoritaria, balanced accuracy maior que 50%, recall minimo de 45% nas duas classes e Brier nao superior ao da base. Nao ha criterios sobre stake ou PnL artificial.

`promote` revalida os hashes e os criterios e grava `approval.json`. Nao altera settings ou inicia o motor. Para ativar, configurar `laya.approved_checkpoint` com o caminho aprovado e reiniciar. O runtime verifica integridade antes de carregar e registra a versao na decisao e na auditoria do contrato.

O alvo proximo fechamento M5 aproxima o contrato real: compra e expiracao ocorrem depois da inferencia. A auditoria guarda tempos/precos reais de entrada e saida e versao do modelo para avaliar essa diferenca. Resultados historicos nao comprovam rentabilidade, e o piso operacional de 1% continua ativo mesmo com Kelly negativo por decisao do operador.
