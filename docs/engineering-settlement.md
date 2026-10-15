# Settlement e fila Redis

Liquidacao assincrona pos-contrato. Codigo: `orchestrator/settlement_*`, `settlement_queue_ops.py`, `post_settlement_*`.

## Fluxo

1. Ordem preenchida → contrato enfileirado
2. Worker polla Redis ate expiry + tolerancia
3. Resolve P&L via stream / `profit_table` se estagnado
4. Atualiza risco (`pending_loss`, linear) e limpa estado
5. `post_settlement` libera proximo ciclo

## Chaves Redis relevantes

| Chave | Uso |
|-------|-----|
| `settlement:queue:priority` | ZSET da fila (score = contract_id) |
| `state:risk:skipped_cycles_counter` | starvation / quality skips |
| `recovery:skip_counter` | contador recovery (limpo pos-ciclo) |
| `session:current:dlambert_unit` | unidade soft recovery |
| `session:current:consecutive_losses_linear` | linear losses |
| `session:current:*` (start balance / target) | alvo de sessao |

Poll minimo da fila: **~2 s** (`settlement_queue_ops`). Tolerancia pos-expiry: settings **600 s** (`settlement_tolerance_window_seconds`; `doctrine_invariants` exige **600**).

## Sintomas

| Log / sintoma | Acao |
|---------------|------|
| Liquidacao estagnada; aguardando profit_table | reconciliar; nao reiniciar a torto e a direito |
| orphans / janela tolerancia | `SETTLE:` — estado reconciliado e OK |
| Spam SETTLE no terminal | `SettlementSpamFilter` / dedupe — nao remover filtro |

## Invariantes

- Enqueue idempotente por `contract_id`
- Atualizacoes ainda abertas nao entram na fila local. Stream, `profit_table` e reconexao compartilham lock por orquestrador e um registro de IDs ja liquidados; saldo, aprendizado e risco sao aplicados uma vez por contrato.
- A compra confirmada entra em `active_contracts` e no risco antes da inscricao de liquidacao. A reconciliação por `profit_table` nao remove o contrato antes de chamar o processador comum.
- A reserva de compra no Redis usa `SET NX` por conta, ativo e vela M5 com TTL de 900 s. Uma falha na reserva impede o envio da ordem; resposta ambigua de compra nao libera a reserva na mesma vela.
- Uma proposta rejeitada antes de `buy` libera apenas a reserva identificada pelo token da propria instancia; o mesmo ativo pode receber nova cotacao na vela, enquanto compra confirmada ou resposta ambigua conserva a reserva.
- Se o `buy` perder o ACK por timeout/desconexao, `unknown_purchase:{conta}:{ativo}` suspende novas compras desse ativo por 900 s, inclusive apos reinicio. Conferir portfolio e `profit_table` antes de interpretar essa janela; o timeout nao prova que a Deriv rejeitou a compra.
- Fila SSOT = Redis **ZSET** `settlement:queue:priority` — **proibido** substituir por Redis Streams, listas ou outro tipo sem mandato explícito + migração testada
- Um consumidor sem o contrato registrado localmente preserva o item no ZSET para a instancia responsavel; somente o dono confirma e remove a liquidacao.
- Nao apagar a fila Redis “para destravar” sem auditar contratos abertos
- Caps e pending continuam soberanos apos WIN operacional parcial
- Redis: AOF everysec + `maxmemory-policy noeviction` (sem drop silencioso de contratos)

Skill: `aether-settlement-debug`. Doutrina CloudOps: [`engineering-devops-cloudops-senior.md`](engineering-devops-cloudops-senior.md).
