import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from src.infrastructure.api.deriv_rest_client import DerivRestClient, DerivRestError


@pytest.mark.asyncio
async def test_bulk_purchase_success():
    client = DerivRestClient(
        rest_base_url="https://api.derivws.com",
        deriv_app_id="1089",
        access_token="ory_at_test",
        timeout_seconds=5,
    )
    payload = json.dumps(
        {
            "data": {
                "transactions": [
                    {
                        "buy_price": "0.35",
                        "contract_id": "99",
                        "payout": "0.66",
                        "purchase_time": 100,
                        "start_time": 100,
                        "shortcode": "CALL_OTC_SPC_0.66_100_220_S0P_0",
                        "account_id": "DOT1",
                        "transaction_id": "1",
                    }
                ]
            }
        }
    ).encode()
    with patch("src.infrastructure.api.deriv_rest_client.read_http_response", return_value=payload) as mock_read:
        tx = await client.bulk_purchase(
            mode="demo",
            account_id="DOT1",
            pat_token="pat_x",
            contract_parameters={
                "amount": 0.35,
                "basis": "stake",
                "contract_type": "CALL",
                "currency": "USD",
                "underlying_symbol": "OTC_SPC",
                "duration": 2,
                "duration_unit": "m",
            },
        )
    assert tx["contract_id"] == "99"
    req = mock_read.call_args.args[0]
    header_names = {k.lower() for k, _ in req.header_items()}
    assert "authorization" not in header_names
    assert req.get_header("Deriv-app-id") == "1089" or req.headers.get("Deriv-App-ID") == "1089"


@pytest.mark.asyncio
async def test_bulk_purchase_rejects_empty_transactions():
    client = DerivRestClient(
        rest_base_url="https://api.derivws.com",
        deriv_app_id="1089",
        access_token="t",
        timeout_seconds=5,
    )
    with (
        patch(
            "src.infrastructure.api.deriv_rest_client.read_http_response",
            return_value=json.dumps({"data": {"transactions": []}}).encode(),
        ),
        pytest.raises(DerivRestError, match="transactions"),
    ):
        await client.bulk_purchase(
            mode="live",
            account_id="R1",
            pat_token="pat",
            contract_parameters={"amount": 1},
        )


@pytest.mark.asyncio
async def test_bulk_purchase_error_branches():
    client = DerivRestClient(
        rest_base_url="https://api.derivws.com",
        deriv_app_id="1089",
        access_token="t",
        timeout_seconds=5,
        max_retries=2,
    )
    with (
        patch(
            "src.infrastructure.api.deriv_rest_client.read_http_response",
            return_value=json.dumps({"data": None}).encode(),
        ),
        pytest.raises(DerivRestError, match="data ausente"),
    ):
        await client.bulk_purchase(mode="demo", account_id="D1", pat_token="p", contract_parameters={})
    with (
        patch(
            "src.infrastructure.api.deriv_rest_client.read_http_response",
            return_value=json.dumps({"data": {"transactions": [None]}}).encode(),
        ),
        pytest.raises(DerivRestError, match="invalida"),
    ):
        await client.bulk_purchase(mode="demo", account_id="D1", pat_token="p", contract_parameters={})
    with (
        patch(
            "src.infrastructure.api.deriv_rest_client.read_http_response",
            return_value=json.dumps({"data": {"transactions": [{"error": "x"}]}}).encode(),
        ),
        pytest.raises(DerivRestError, match="conta falhou"),
    ):
        await client.bulk_purchase(mode="demo", account_id="D1", pat_token="p", contract_parameters={})


def test_request_pat_body_http_and_url_errors():
    client = DerivRestClient(
        rest_base_url="https://api.derivws.com",
        deriv_app_id="1089",
        access_token="t",
        timeout_seconds=5,
        max_retries=2,
    )
    http_exc = urllib.error.HTTPError("https://x", 400, "bad", hdrs=None, fp=None)
    http_exc.read = MagicMock(return_value=b"nope")
    with (
        patch("src.infrastructure.api.deriv_rest_client.read_http_response", side_effect=http_exc),
        pytest.raises(DerivRestError, match="HTTP 400"),
    ):
        client._request_pat_body("POST", "/bulk", body={"a": 1})
    with (
        patch(
            "src.infrastructure.api.deriv_rest_client.read_http_response",
            side_effect=[urllib.error.URLError("down"), json.dumps({"ok": True}).encode()],
        ),
        patch("src.infrastructure.api.deriv_rest_client.time.sleep"),
    ):
        assert client._request_pat_body("POST", "/bulk", body={"a": 1}) == {"ok": True}
    with (
        patch(
            "src.infrastructure.api.deriv_rest_client.read_http_response",
            side_effect=urllib.error.URLError("down"),
        ),
        patch("src.infrastructure.api.deriv_rest_client.time.sleep"),
        pytest.raises(DerivRestError, match="falhou"),
    ):
        client._request_pat_body("POST", "/bulk", body={"a": 1})
    with (
        patch("src.infrastructure.api.deriv_rest_client.read_http_response", return_value=b"[1,2,3]"),
        pytest.raises(DerivRestError, match="JSON invalida"),
    ):
        client._request_pat_body("POST", "/bulk", body={"a": 1})
