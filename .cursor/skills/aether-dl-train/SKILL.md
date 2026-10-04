---
name: aether-dl-train
description: Diagnostica treino e inferencia TCN M5, checkpoint tecnico, calibracao e meta LightGBM.
---

# Treino e inferencia DL

1. Conferir `config/settings.json`: TCN, lookback 32, 25.000 velas M5, label `spot_forward`, horizonte de uma vela, contrato de cinco minutos.
2. Conferir a extração causal de 14 features e a geometria do checkpoint: pesos, normalização, feature_dim, lookback, granularidade, label e horizonte.
3. Executar `train.py` e depois `check_dl_checkpoint.py`; o segundo script verifica somente integridade e geometria.
4. Tratar benchmark linear, `val_accuracy`, Brier, calibração, `label_call_frac`, `pred_call_frac` e `minority_recall` como diagnóstico do treino. Comparar a acuracia com o classificador majoritario; o label por barreira nao reproduz o resultado confirmado do contrato. Não há qualificação estatística de deploy.
5. Com checkpoint técnico válido, DEMO e REAL usam o teto inicial de 1% da banca por ordem, inclusive na recuperação `cover_l0`; sem checkpoint válido, a operação é bloqueada.
6. `online_training=false`: o runtime usa o checkpoint local salvo pelo treino separado. Mudança de label, horizonte, lookback, granularidade ou features exige retreino.
7. Meta LightGBM e loss-classifier são treinados separadamente; após mudar sidecars, executar `make docker-rebuild` sem apagar `data/dl`.
8. Inferência TCN fica no host; trabalhos CUDA pesados não devem bloquear o event loop do WebSocket.
9. Para comparar a perda ponderada com BCE sem alterar o checkpoint, rodar `python app/scripts/operations/compare_dl_losses.py --bars 5000 --epochs 40 --seed 42`. Ler também `full_window_accuracy` e `full_window_brier`, pois o treino sparse não representa todos os ciclos.
10. `contract_label_audit` usa velas M5 e somente lucro `broker` confirmado. A direção da vela é proxy; contratos atribuídos escassos não bastam para treinar ou alegar EV positivo.

Doc: `docs/engineering-deep-learning.md`.
