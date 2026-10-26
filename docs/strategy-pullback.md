# Setup institucional M15

## Sinal operacional

O indicador consulta o catalogo publico e as velas OHLC da Deriv. O motor compra apenas sinais `senior_institutional` de M15 da versao configurada, com vela fechada atual e entrada ate 60 segundos apos a abertura da proxima vela. O contrato Rise/Fall dura 15 minutos. H1 e D1 sao pesquisa; M5 do stream legado nao define nem altera o lado.

O setup usa 128 velas de M15 para aquecer as medias e exige, para CALL:

1. EMA20 acima da EMA50, EMA20 crescente nas ultimas tres transicoes e EMA50 ascendente.
2. Vela de recuo contraria a tendencia, com minima a ate 0,5 ATR14 da EMA20 e fechamento acima da EMA50; confirmacao bullish acima da maxima do recuo, corpo de 0,03 a 1,5 ATR14 e fechamento a ate 0,75 ATR14 da EMA20.
3. Fechamento acima da referencia de preco da sessao UTC e abaixo de sua banda superior de 1,5 desvio padrao.
4. Fair Value Gap bullish efetivo em velas anteriores com recuo que testa sua faixa.
5. Vela de confirmacao nas horas BRT 03-07, 10-12 ou 14-15. Lista ausente usa essas horas; lista vazia bloqueia o sinal.

PUT espelha as condicoes. A referencia da sessao e a media dos precos tipicos `(high + low + close) / 3` das velas desde 00:00 UTC, com bandas de desvio padrao. O OHLC publico nao fornece volume para esta rota: **essa medida nao e VWAP por volume**. Menos de duas velas na sessao produz `SEM SINAL`. Nao se usa preco de outra fonte para completar a serie da Deriv.

Lacunas intradiarias, velas duplicadas conflitantes e dado atrasado produzem `SEM SINAL`. Para OTC, uma pausa entre dias UTC de ate 72 horas pode separar sessoes; as velas sao mantidas com seus horarios originais, sem interpolacao, e a vela de confirmacao precisa ser contigua a anterior.

## Treino e evidencia

`make app-train` calcula apenas um **proxy direcional de velas** para M15, H1 e D1 e mostra ranking diagnostico por ativo. Nao muda `config/settings.json`, nao escolhe o ativo operacional e nao marca a versao como `ativo`. OHLC externo, quando consultado em pesquisas separadas, nao e concatenado ao OHLC da Deriv. Taxa de acerto ou EV do proxy nao representa resultado de contrato.

O piloto DEMO da versao nova aceita ate 200 propostas validas com stake base de 1% da banca disponivel, mesmo sem EV positivo. Somente contratos unicos liquidados pelo broker e atribuidos ao mesmo ativo, periodo e versao compoem a amostra. O registro muda de `piloto` para `ativo` apenas com 200 compras e liquidacoes e limite inferior unilateral de 95% do retorno liquido medio positivo; caso contrario, vai para `reprovado`.

REAL exige adicionalmente essa evidencia no motor, posterior Beta(0,5; 0,5), haircut, EV positivo e margem de 0,01 acima do break-even na cotacao final. A compra conserva o lado do sinal aprovado. Sem sinal atual ou cotacao valida, nao ha ordem.

## Risco e auditoria

A stake base e 1% da banca disponivel. A recuperacao 1/1 usa o payout liquido cotado e respeita 5% por ordem; perdas pendentes nao geram sinal. Reservas por conta/ativo/vela, deduplicacao por `contract_id` e reconciliacao de compras sem ACK continuam obrigatorias. Nao ha stop loss ou teto acumulado de perda.
