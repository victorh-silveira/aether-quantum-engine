"""Testes de integracao com stack Docker (opcional)."""

import pytest


@pytest.mark.docker
@pytest.mark.asyncio
async def test_infra_stack_placeholder():
    """Marcador reservado para validacao E2E com docker compose."""
    pytest.skip("Execute com docker compose up e servicos em localhost")
