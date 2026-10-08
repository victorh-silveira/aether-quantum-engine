# Treino e inferência Deep Learning

## Runtime atual

O fluxo ativo usa TCN em velas M5 do índice `1HZ75V`. A configuração em `config/settings.json` define 25.000 barras de treino, lookback 32, 14 features, label `spot_forward`, horizonte de uma vela e contrato Rise/Fall de 5 minutos. D1 fornece contexto macro ao motor; o treino TCN lean carrega apenas M5. O motor usa o checkpoint local produzido por `launch-train`; `online_training=false` impede retreino TCN durante a sessão.

O label `spot_forward` usa a direção da próxima vela fechada. Esse label não equivale ao resultado confirmado de um contrato Rise/Fall. O treino usa validação interna para ajuste, calibração e detecção de colapso de classe. O benchmark linear compara uma regressão logística a um classificador majoritário nesse mesmo recorte temporal; é apenas telemetria. **Não há qualificação estatística da TCN nem promoção baseada nessas métricas.** Um checkpoint com pesos, normalização e geometria incompatíveis não é carregado. O verificador `app/scripts/operations/check_dl_checkpoint.py` confere esses requisitos após o treino.

A perda ativa de classificação é BCE simétrica (`asymmetric_payout_loss=false`). Pesar CALL por 0,85 e PUT por 1,0 desloca a probabilidade ótima para PUT mesmo com labels 50/50; o payout deve entrar no cálculo de EV da proposta, não na probabilidade CALL. Na seleção interna de épocas, estados sem colapso de classe têm prioridade; se todos colapsarem, o melhor estado colapsado permanece como fallback técnico e o diagnóstico continua explícito. Isso não qualifica o modelo nem demonstra vantagem. Após mudar a perda, retreine TCN e meta.

Com checkpoint técnico válido, DEMO e REAL usam a mesma inferência e a mesma abertura. O teto inicial de stake é **1% da banca por ordem**, inclusive em recuperação `cover_l0`; outros limites podem reduzi-lo. Sem checkpoint válido, a execução fica bloqueada. A validação do treino não demonstra vantagem preditiva.

## Sequência de treino

1. `launch-train` prepara o ambiente e treina o loss-classifier.
2. O TCN ajusta pesos em velas M5 com splits temporais para treino, validação e calibração. O horizonte do label e a duração do contrato são de 300 segundos.
3. O checkpoint local guarda pesos, normalização, arquitetura, lookback, granularidade, label e horizonte. O verificador pós-treino falha se o arquivo estiver ausente, corrompido ou incompatível com o SSOT.
4. O treino meta LightGBM usa o checkpoint como teacher e dados disponíveis de mercado. Os sidecars continuam independentes da validade técnica do checkpoint TCN.

Alterar timeframe, lookback, horizonte, features ou semântica do label exige novo treino. Manter 14 colunas não garante compatibilidade semântica após mudar a transformação de indicadores.

## Diagnóstico e experimento offline

Os resultados obtidos no experimento anterior em M1 com `triple_barrier` pertencem à configuração anterior e não descrevem a qualidade do novo checkpoint M5. A acurácia de treino e validação, Brier, ECE e o benchmark linear servem para diagnosticar ajuste e colapso de classe. Essas métricas não demonstram EV positivo de contratos.

`app/scripts/operations/compare_dl_losses.py` compara perda ponderada por payout e BCE com a mesma inicialização, amostras e split temporal. A ferramenta lê a granularidade ativa de `settings.json`, não exporta checkpoint e exige histórico M5 contínuo no Timescale.

```bash
python app/scripts/operations/compare_dl_losses.py --bars 5000 --epochs 40 --seed 42
```

É necessário repetir a comparação com dados M5 antes de concluir qual perda se ajusta melhor. Nenhum resultado antigo em M1 deve ser atribuído à configuração M5.

No retreino M5 de 04/10/2026, a TCN usou 25.000 velas, lookback 32 e 300 épocas. A validação temporal teve acurácia 0,505, abaixo da maioria de 0,508, Brier 0,316 e ECE 0,216; o treino teve acurácia 0,544. O checkpoint passou na verificação técnica, e o meta foi treinado sobre 5.000 velas M5, mas esses números não demonstram sinal preditivo ou EV positivo.

No log posterior enviado pelo operador, o treino com perda assimétrica produziu `val_acc=0.509` contra maioria `0.508`, `pred_call=0.01` e `minority_rec=0.01`. Esse checkpoint continua tecnicamente legível, mas os diagnósticos indicam colapso para PUT. O ajuste da perda corrige o viés matemático da função objetivo; não garante que as features M5 contenham sinal previsível ou que apareçam trades com EV positivo.

## Inferência e decisão

A inferencia na abertura M5 usa a ultima vela fechada como fim da janela de 32 barras. O label `spot_forward` desse ponto representa a vela M5 corrente, na qual o contrato de 300 segundos e aberto. Se o buffer ja inclui a vela corrente em formacao, essa linha e excluida do tensor TCN e do eventual treino deferido. O snapshot com tick live continua disponivel para telemetria e verificacoes de mercado. `model_input_forming_excluded` registra a exclusao por ciclo. Usar a vela em formacao no tensor desloca a previsao para a vela posterior ao contrato.

A inferencia carrega 768 velas M5 antes de montar as 32 entradas da TCN. A janela anterior de 384 barras alterava features da mesma sequencia em relacao ao calculo com historico longo do treino: em 1.200 velas locais, a diferenca maxima chegou a 0,882 na feature `norm_frac_diff`. Com 768 barras, a mesma sequencia ficou identica ao historico longo nesse recorte. O startup de inferencia usa `inference_history_bars` mais o warmup configurado; a mudanca nao exige retreino porque recupera a transformacao usada no treino, mas nao comprova vantagem preditiva.

O `[CLUSTER]` mostra `input=closed` quando excluiu a vela em formacao e `input=latest` quando a ultima barra recebida ja estava fechada. A projecao `[NEXT_MOVE]` so e emitida se a cabeca auxiliar de regressao estiver treinada (`aux_regression_weight>0`); com peso zero, seus valores nao sao previsoes validas.

A TCN estima `P(CALL)` calibrada. CALL é o lado inicial quando Cal ≥ 0,5; caso contrário, PUT. A regra usa a probabilidade final apos o clamp, mesmo quando ela atravessa 0,5 ou quando o raw recebe a marca `raw_extreme`; a marca nao inverte o lado. Os limiares de confiança adicionais e a faixa neutra alimentam telemetria e filtros, mas não mudam essa regra básica. O loss-classifier pode inverter o lado após aprendizado com contratos reais; não se deve inferir vantagem de uma inversão arbitrária. Os vetos de mercado, a cotação e o edge da proposta final podem impedir a compra.

O cálculo de features e normalização precisa ser causal: acrescentar candles futuros não pode alterar indicadores já calculados. Para inferência CUDA, o modelo roda no host, e trabalhos pesados são deslocados para não bloquear o event loop do WebSocket.

## Captura de mercado e contratos

`infra.timescale.capture_enabled=true` liga o writer Timescale quando o motor está ativo. A captura registra ticks e contratos confirmados e mantém proveniência do resultado (`broker`, `profit_table` ou `inferred_rest`). A view `contract_label_audit` cruza contratos Rise/Fall de aproximadamente 300 segundos com resultado `broker` confirmado e compara a direção da próxima vela M5 (`close` versus `open`) com o lucro confirmado. A direção da vela é um proxy de mercado, não o resultado do contrato. A view tambem mostra o deslocamento de spot do contrato e da vela, a fase de entrada M5 e a duracao observada. Velas ausentes permanecem `NULL`. A migração `009_contract_label_audit_m5.sql` atualiza a view em volumes existentes; `make docker-timescale-lifecycle` a reaplica. A view `contract_model_outcomes` agrega contratos por versão do checkpoint, símbolo, tipo e conta. Nenhuma dessas views qualifica modelos ou altera o teto de stake.

Na observação anterior em 04/10/2026 havia somente 3 contratos `broker` atribuídos, insuficientes para concluir que existe EV positivo. O histórico M1 coletado naquela ocasião não constitui uma avaliação M5.

Para coletar ticks públicos sem abrir trades, use `python app/scripts/operations/collect_public_ticks.py`. A coleta não faz backfill automático; a retenção de ticks é de 30 dias. Ao comparar labels com resultados reais, verifique timestamps, payout e fonte do settlement antes de interpretar diferenças.

## Operação

```bash
python train.py
python app/scripts/operations/check_dl_checkpoint.py
python run.py
```

O launcher WSL `app/scripts/wsl/launch-train-wsl.sh` executa o pipeline completo. Após alterar os sidecars, `make docker-rebuild` recarrega meta e loss-classifier sem apagar o checkpoint local.

Consulte [arquitetura](arquitetura.md), [settings](engineering-settings-ssot.md) e [infraestrutura](infra-docker.md).
