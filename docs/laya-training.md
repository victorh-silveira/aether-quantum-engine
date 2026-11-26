# Treinamento temporal Laya para R_10 M5

O fluxo offline usa Laya classico 0.5.5, `vegaml=False`, sem executor de ordens. Preparar os dados e validar tokens funciona em CPU; treinar exige CUDA com pelo menos 8 GiB livres. Nao existe fallback de treinamento pesado em CPU.

## Comandos no WSL com Conda deriv-api

```bash
python -m tools.laya_workflow prepare --data .cache/laya-r10-gpu-20261126
python -m tools.laya_workflow train --data .cache/laya-r10-gpu-20261126 --run .cache/laya-r10-gpu-run --dry-run
python -m tools.laya_workflow train --data .cache/laya-r10-gpu-20261126 --run .cache/laya-r10-gpu-run
python -m tools.laya_workflow evaluate --data .cache/laya-r10-gpu-20261126 --run .cache/laya-r10-gpu-run
python -m tools.laya_workflow promote --data .cache/laya-r10-gpu-20261126 --run .cache/laya-r10-gpu-run
```

Substituir `python` pelo executavel do ambiente Conda quando necessario. Copiar o diretorio de dados e os artefatos do checkpoint base para o host GPU; os manifestos identificam a revisao e os arquivos usados. `--help` descreve os argumentos. `evaluate --device cpu` e permitido explicitamente, mas pode ser lento.

## Dados e ausencia de vazamento

`prepare_request.json` fixa o corte na primeira tentativa. O coletor reutiliza parquet local R_10/300s e consulta paginas publicas da Deriv, sem PAT. Precos sinteticos ou interpolados sao proibidos: lacunas confirmadas pela API excluem exemplos cuja janela de 120 barras as atravesse. Historico anterior esgotado interrompe a preparacao.

Sao 90 dias mais 120 barras de aquecimento. A divisao por tempo e 60% treino, 10% calibracao, 10% selecao e 20% teste; horizontes que alcancem a proxima divisao sao purgados. Empates de fechamento sao contados e excluidos do alvo binario. O contexto inclui 12 velas, o mesmo serializador e os mesmos indicadores causais do runtime. Os hashes vinculam perguntas, configuracoes sanitizadas e dados; mudancas exigem novo dataset.

O dry-run verifica todos os tokens, sem pesos. O orcamento inclui pergunta e opcoes e nao pode ultrapassar a capacidade do encoder. Os dados de teste sao inspecionados apenas para integridade e comprimento nessa etapa.

## Treinamento GPU e selecao

Treinar seeds 17, 42 e 73 com RLCD, quatro epocas, microbatch 1, acumulacao 64, encoder LR 0,000025 e head LR 0,0001; usar precisao mista e gradient checkpointing. Cada seed salva `checkpoint_latest` ao final de cada epoca para preservar pesos de diagnostico em caso de falha. O comando recusa sobrescrever uma execucao anterior; iniciar outra exige diretorio novo.

A calibracao usa exclusivamente a faixa temporal reservada, sem sorteio do holdout padrao da biblioteca. Selecionar maior acuracia na faixa de selecao, desempatar pelo menor Brier e depois pela ordem das seeds. Salvar pesos, tokenizer, perguntas, temperaturas, identidade da base e manifesto de treinamento.

## Teste final, promocao e limites

`evaluate` exige candidato previamente selecionado e consome o teste uma unica vez, marcado por `test_consumed.json`. Nao treinar novos candidatos nesse dataset depois disso. Uma falha de avaliacao conserva o marcador e impede ajuste oportunista; investigar e preparar um novo dataset independente.

Aprovacao exige simultaneamente: melhora de pelo menos 1 ponto percentual de acuracia sobre a base, acuracia superior a classe majoritaria, balanced accuracy maior que 50%, recall minimo de 45% nas duas classes e Brier nao superior ao da base. Nao ha criterios sobre stake ou PnL artificial.

`promote` revalida os hashes e os criterios e grava `approval.json`. Nao altera settings ou inicia o motor. Para ativar, configurar `laya.approved_checkpoint` com o caminho aprovado e reiniciar. O runtime verifica integridade antes de carregar e registra a versao na decisao e na auditoria do contrato.

O alvo proximo fechamento M5 aproxima o contrato real: compra e expiracao ocorrem depois da inferencia. A auditoria guarda tempos/precos reais de entrada e saida e versao do modelo para avaliar essa diferenca. Resultados historicos nao comprovam rentabilidade, e o piso operacional de 1% continua ativo mesmo com Kelly negativo por decisao do operador.
