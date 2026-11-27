# Treinamento temporal Laya para R_10 M5

O fluxo offline usa Laya classico 0.5.5, `vegaml=False`, sem executor de ordens. Preparar os dados e validar tokens funciona em CPU; treinar exige CUDA com pelo menos 3 GiB livres. O perfil `auto` usa encoder congelado (`head`) em GPUs de ate 12 GiB, incluindo a RTX 4060 de 8 GiB; em GPUs maiores usa `full`, que exige ao menos 12 GiB livres. Nao existe fallback de treinamento pesado em CPU.

## Comandos no WSL com Conda deriv-api

`make help` mostra comandos essenciais e descricoes objetivas em uma frase nos grupos App, CI/CD e Docker. CI/CD possui Local, Pre-commit e Workflows; metadados e tutoriais ficam nesta documentacao, fora do menu. Os alvos avancados continuam disponiveis diretamente. Instalacao e diagnostico sao etapas de configuracao; treinamento e uma operacao separada da rotina diaria.

Para o ambiente ja instalado e usando o checkpoint configurado:

```bash
make docker-up
make app-run
```

`docker-up` inclui healthchecks e smoke checks; `app-run` verifica CUDA antes de iniciar. A configuracao de credenciais e conta deve estar concluida no ambiente. Instalar dependencias ou hooks, executar CI/CD e treinar novamente nao sao requisitos de cada inicializacao.

Para preparar um ambiente novo e treinar um candidato, seguir as etapas abaixo. Omitir app-prepare quando o dataset ja existir.

```bash
make app-install
make app-cuda-install
make app-cuda-check
make app-cuda-train-check
make app-prepare
make app-train-dry-run
make app-train
make app-evaluate
make app-promote
```

Apos promocao aprovada, configurar `laya.approved_checkpoint` em `config/settings.json` com o caminho retornado pelo promote. Entao executar `make docker-up` e `make app-run`. Sem esse ajuste o motor continua usando o checkpoint anteriormente configurado.

Pre-commit e desenvolvimento: instalar hooks com `make app-pre-commit` na configuracao inicial; usar `make app-pre-commit-run` antes do commit. Para diagnosticar uma etapa isolada do CI local, usar app-lint, app-validate, app-security, app-test e app-build. O CI/CD remoto e definido em `.github/workflows/ci.yml`: push na main, PR para main ou acionamento manual; a publicacao de versao ocorre na main apos CI aprovado.

app-clean, docker-reset e docker-clean sao operacoes de manutencao; nao fazem parte da sequencia de inicializacao.

O Makefile resolve o Python do Conda `deriv-api` e usa `LAYA_DATA=.cache/laya-r10-gpu-20261126`, `LAYA_RUN=.cache/laya-r10-gpu-run-fast`, `LAYA_TRAIN_PROFILE=auto`, `LAYA_TRAIN_PRESET=fast` e `LAYA_WARM_START=.cache/laya-r10-gpu-run/seed-17/checkpoint_latest` por padrao. Alterar diretorios com `make app-train LAYA_DATA="caminho dos dados" LAYA_RUN="nova execucao"`. Preparacao de dados existentes nao deve ser repetida: o manifesto ja fixa o corte. `make app-evaluate LAYA_DEVICE=cpu` permite avaliacao explicitamente em CPU. `LAYA_ARGS` acrescenta argumentos de uma etapa, como `--cutoff` no prepare ou `--resume` para retomar uma execucao interrompida.

`app-cuda-check` prova inferencia real no device CUDA; `app-cuda-train-check` prova backward e atualizacao de parametros em memoria, sem gravar pesos ou ordens. `app-run` verifica CUDA antes de executar run.py, que continua usando o checkpoint base ou o caminho aprovado configurado em settings. Promover nao ativa automaticamente um checkpoint.

Copiar dados e checkpoint base para outro host GPU preserva os manifestos. `make app-train LAYA_TRAIN_PROFILE=full` solicita treinamento completo e exige a margem de memoria correspondente.

## Dados e ausencia de vazamento

`prepare_request.json` fixa o corte na primeira tentativa. O coletor reutiliza parquet local R_10/300s e consulta paginas publicas da Deriv, sem PAT. Precos sinteticos ou interpolados sao proibidos: lacunas confirmadas pela API excluem exemplos cuja janela de 120 barras as atravesse. Historico anterior esgotado interrompe a preparacao.

Sao 90 dias mais 120 barras de aquecimento. A divisao por tempo e 60% treino, 10% calibracao, 10% selecao e 20% teste; horizontes que alcancem a proxima divisao sao purgados. Empates de fechamento sao contados e excluidos do alvo binario. O contexto inclui 12 velas, o mesmo serializador e os mesmos indicadores causais do runtime. Os hashes vinculam perguntas, configuracoes sanitizadas e dados; mudancas exigem novo dataset.

O dry-run verifica todos os tokens, sem pesos. O orcamento inclui pergunta e opcoes e nao pode ultrapassar a capacidade do encoder. Os dados de teste sao inspecionados apenas para integridade e comprimento nessa etapa.

## Treinamento GPU, presets e selecao

O treinamento suporta `--preset fast` (padrao via `LAYA_TRAIN_PRESET=fast`: seed 17, 2 epocas totais) e `--preset complete` (modo explicito com seeds 17, 42 e 73, 4 epocas por seed). Em ambos os presets sao preservados os 90 dias de dados, as divisoes temporais 60/10/10/20, a calibracao externa e todos os criterios de avaliacao e promocao.

No modo `fast`, se `LAYA_WARM_START` (`.cache/laya-r10-gpu-run/seed-17/checkpoint_latest`) existir, o fluxo valida estrutura, pesos, tokenizer e compatibilidade de contexto antes de carregar, registra 1 epoca herdada, reinicia o otimizador (pois o checkpoint legado guarda apenas pesos) e executa apenas a epoca restante, preservando integralmente `.cache/laya-r10-gpu-run`. Se o caminho legado estiver ausente, inicia 2 epocas a partir do modelo base; se estiver presente mas incompativel, encerra com diagnostico.

No perfil `head`, o fluxo mede lotes de 1, 2 e 4 exemplos exclusivamente da particao de treino antes do treinamento, escolhe o maior lote com margem minima de 1 GiB (recuando em caso de OOM) e ajusta a acumulacao para 64, 32 ou 16, mantendo o lote efetivo de 64 exemplos, os learning rates (encoder LR 0,000025 e head LR 0,0001) e o objetivo RLCD com precisao mista e gradient checkpointing. Apos o benchmark de lote, pesos da cabeca, buffers e RNGs sao restaurados. Os exemplos de treino sao tokenizados uma vez por execucao e reutilizados entre epocas.

O acompanhamento exibe separadamente as fases de preparacao, treinamento, calibracao, selecao e gravacao, com epoca, amostras processadas/total, exemplos por segundo, tempo decorrido e ETA (exibindo `calculando` antes de haver amostras suficientes). Um lock exclusivo impede executar dois treinamentos simultaneos na GPU. Os novos checkpoints salvam tambem epoca, posicao, otimizador, scheduler, GradScaler e estados aleatorios para retomada exata via `make app-train LAYA_ARGS=--resume`, distinguindo nos manifestos (`run.json`, `training.json`, `selection.json`) `exact_resume` de `legacy_warm_start` e `fresh`, alem de registrar hash e origem do checkpoint, reinicio do otimizador, perfil, lote e identidade do dataset.

A calibracao usa exclusivamente a faixa temporal reservada, sem sorteio do holdout padrao da biblioteca. Selecionar maior acuracia na faixa de selecao, desempatar pelo menor Brier e depois pela ordem das seeds. Salvar pesos, tokenizer, perguntas, temperaturas, identidade da base e manifesto de treinamento.

## Teste final, promocao e limites

`evaluate` exige candidato previamente selecionado e consome o teste uma unica vez, marcado por `test_consumed.json`. Nao treinar novos candidatos nesse dataset depois disso. Uma falha de avaliacao conserva o marcador e impede ajuste oportunista; investigar e preparar um novo dataset independente.

Aprovacao exige simultaneamente: melhora de pelo menos 1 ponto percentual de acuracia sobre a base, acuracia superior a classe majoritaria, balanced accuracy maior que 50%, recall minimo de 45% nas duas classes e Brier nao superior ao da base. Nao ha criterios sobre stake ou PnL artificial.

`promote` revalida os hashes e os criterios e grava `approval.json`. Por padrao, exige aprovacao completa no teste reservado; para testes experimentais em conta demo, `make app-promote LAYA_ARGS=--allow-unapproved-override` permite override explicito do operador mantendo `approved: false`, registrando `operator_override: true` e preservando toda a validacao de integridade dos hashes. Nao altera settings ou inicia o motor. Para ativar, configurar `laya.approved_checkpoint` com o caminho aprovado e reiniciar. O runtime verifica integridade antes de carregar e registra a versao na decisao e na auditoria do contrato.

O alvo proximo fechamento M5 aproxima o contrato real: compra e expiracao ocorrem depois da inferencia. A auditoria guarda tempos/precos reais de entrada e saida e versao do modelo para avaliar essa diferenca. Resultados historicos nao comprovam rentabilidade, e o piso operacional de 1% continua ativo mesmo com Kelly negativo por decisao do operador.
