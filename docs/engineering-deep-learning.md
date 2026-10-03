# Treino e inferência Deep Learning

## Runtime atual

O fluxo ativo usa TCN em velas M5 do índice `1HZ75V`. A configuração em `config/settings.json` define 25.000 barras de treino, lookback 32, 14 features, label `spot_forward`, horizonte de uma vela e contrato Rise/Fall de 5 minutos. D1 fornece contexto macro. O motor usa o checkpoint local produzido por `launch-train`; `online_training=false` impede retreino TCN durante a sessão.

O label por fechamento é uma aproximação da direção do contrato. O treino usa validação interna para ajuste, calibração e detecção de colapso de classe. **Não há qualificação estatística da TCN, avaliação diagnóstica de liquidação por closes M5 nem promoção da TCN baseada nessas métricas.** Um checkpoint com pesos, normalização e geometria incompatíveis não é carregado. O verificador `app/scripts/operations/check_dl_checkpoint.py` confere esses requisitos após o treino.

Com checkpoint técnico válido, DEMO e REAL usam a mesma inferência e a mesma abertura. O teto inicial de stake é **1% da banca por ordem**, inclusive em recuperação `cover_l0`; outros limites podem reduzi-lo. Sem checkpoint válido, a execução fica bloqueada. A validação do treino não demonstra vantagem preditiva.

## Sequência de treino

1. `launch-train` prepara o ambiente e treina o loss-classifier.
2. O TCN ajusta pesos em velas M5 com splits temporais para treino, validação e calibração. O label continua alinhado à duração de 5 minutos.
3. O checkpoint local guarda pesos, normalização, arquitetura, lookback, granularidade, label e horizonte. O verificador pós-treino falha se o arquivo estiver ausente, corrompido ou incompatível com o SSOT.
4. O treino meta LightGBM usa o checkpoint como teacher e dados disponíveis de mercado. Os sidecars continuam independentes da validade técnica do checkpoint TCN.

Alterar timeframe, lookback, horizonte, features ou semântica do label exige novo treino. Manter 14 colunas não garante compatibilidade semântica após mudar a transformação de indicadores.

## Inferência e decisão

A TCN estima `P(CALL)` calibrada. CALL é o lado inicial quando Cal ≥ 0,5; caso contrário, PUT. Os limiares de confiança adicionais e a faixa neutra alimentam telemetria e filtros, mas não mudam essa regra básica. O loss-classifier pode inverter o lado após aprendizado com contratos reais; não se deve inferir vantagem de uma inversão arbitrária. Os vetos de mercado, a cotação e o edge da proposta final podem impedir a compra.

O cálculo de features e normalização precisa ser causal: acrescentar candles futuros não pode alterar indicadores já calculados. Para inferência CUDA, o modelo roda no host, e trabalhos pesados são deslocados para não bloquear o event loop do WebSocket.

## Captura de mercado e contratos

`infra.timescale.capture_enabled=true` liga o writer Timescale quando o motor está ativo. A captura registra ticks e contratos confirmados e mantém proveniência do resultado (`broker`, `profit_table` ou `inferred_rest`). A view `contract_label_audit` cruza apenas contratos Rise/Fall com resultados confirmados; spots ausentes permanecem `NULL`. A view `contract_model_outcomes` agrega contratos por versão do checkpoint, símbolo, tipo e conta. Essas views servem à auditoria operacional e não qualificam automaticamente modelos nem alteram o teto de stake.

Para coletar ticks públicos sem abrir trades, use `python app/scripts/operations/collect_public_ticks.py`. A coleta não faz backfill automático; a retenção de ticks é de 30 dias. Ao comparar labels com resultados reais, verifique timestamps, payout e fonte do settlement antes de interpretar diferenças.

## Operação

```bash
python train.py
python app/scripts/operations/check_dl_checkpoint.py
python run.py
```

O launcher WSL `app/scripts/wsl/launch-train-wsl.sh` executa o pipeline completo. Após alterar os sidecars, `make docker-rebuild` recarrega meta e loss-classifier sem apagar o checkpoint local.

Consulte [arquitetura](arquitetura.md), [settings](engineering-settings-ssot.md) e [infraestrutura](infra-docker.md).
