import asyncio
import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from src.infrastructure.api.deriv_rest_client import (
    DerivAccount,
    DerivRestClient,
    DerivRestError,
    _account_type_for_mode,
    select_account,
)


def test_account_type_for_mode():
    assert _account_type_for_mode("demo") == "demo"
    assert _account_type_for_mode("live") == "real"
    assert _account_type_for_mode("real") == "real"


def test_select_account_by_mode_and_override():
    accounts = [
        DerivAccount("D1", 100.0, "demo", "active", "USD"),
        DerivAccount("R1", 200.0, "real", "active", "USD"),
    ]
    assert select_account(accounts, "demo").account_id == "D1"
    assert select_account(accounts, "live").account_id == "R1"
    assert select_account(accounts, "demo", "R1").account_id == "R1"
    with pytest.raises(DerivRestError):
        select_account(accounts, "demo", "MISSING")


def test_select_account_inactive_fallback_and_empty():
    inactive = [DerivAccount("D2", 1.0, "demo", "disabled", "USD")]
    assert select_account(inactive, "demo").account_id == "D2"
    with pytest.raises(DerivRestError):
        select_account([DerivAccount("X", 1.0, "real", "active", "USD")], "demo")


@pytest.mark.asyncio
async def test_open_trading_session():
    client = DerivRestClient(
        rest_base_url="https://api.derivws.com",
        deriv_app_id="1089",
        access_token="ory_at_test",
        timeout_seconds=5,
    )
    accounts_payload = json.dumps(
        {
            "data": [
                {
                    "account_id": "DOT1",
                    "balance": 500.5,
                    "account_type": "demo",
                    "status": "active",
                    "currency": "USD",
                }
            ]
        }
    ).encode()
    otp_payload = json.dumps({"data": {"url": "wss://api.derivws.com/trading/v1/options/ws/demo?otp=abc"}}).encode()

    def fake_read(req, timeout):
        if req.full_url.endswith("/otp"):
            return otp_payload
        return accounts_payload

    with patch("src.infrastructure.api.deriv_rest_client.read_http_response", side_effect=fake_read):
        session = await client.open_trading_session("demo")
    assert session.balance == 500.5
    assert "otp=abc" in session.ws_url


@pytest.mark.asyncio
async def test_post_otp_issues_fresh_ws_url():
    client = DerivRestClient(
        rest_base_url="https://api.derivws.com",
        deriv_app_id="1089",
        access_token="ory_at_test",
        timeout_seconds=5,
    )
    otp_payload = json.dumps({"data": {"url": "wss://api.derivws.com/trading/v1/options/ws/demo?otp=new"}}).encode()

    with patch(
        "src.infrastructure.api.deriv_rest_client.read_http_response",
        return_value=otp_payload,
    ):
        url = await client.post_otp("DOT1")
    assert "otp=new" in url


def test_list_accounts_invalid_payload():
    client = DerivRestClient(
        rest_base_url="https://api.derivws.com",
        deriv_app_id="1089",
        access_token="t",
        timeout_seconds=5,
    )
    with (
        patch(
            "src.infrastructure.api.deriv_rest_client.read_http_response",
            return_value=json.dumps({"meta": {}}).encode(),
        ),
        pytest.raises(DerivRestError),
    ):
        asyncio.run(client.list_accounts())


def test_list_accounts_skips_bad_rows_and_empty():
    client = DerivRestClient(
        rest_base_url="https://api.derivws.com",
        deriv_app_id="1089",
        access_token="t",
        timeout_seconds=5,
    )
    payload = json.dumps({"data": ["x", {"no_id": 1}]}).encode()
    with (
        patch("src.infrastructure.api.deriv_rest_client.read_http_response", return_value=payload),
        pytest.raises(DerivRestError, match="Nenhuma conta"),
    ):
        asyncio.run(client.list_accounts())


def test_request_http_and_json_errors():
    client = DerivRestClient(
        rest_base_url="https://api.derivws.com",
        deriv_app_id="1089",
        access_token="t",
        timeout_seconds=5,
    )
    err = urllib.error.HTTPError("u", 401, "n", {}, None)
    err.read = MagicMock(return_value=b"denied")
    with (
        patch("src.infrastructure.api.deriv_rest_client.read_http_response", side_effect=err),
        pytest.raises(DerivRestError, match="HTTP 401"),
    ):
        asyncio.run(client._request("GET", "/trading/v1/options/accounts"))
    with (
        patch(
            "src.infrastructure.api.deriv_rest_client.read_http_response",
            side_effect=urllib.error.URLError("offline"),
        ),
        patch("src.infrastructure.api.deriv_rest_client.time.sleep", return_value=None),
        pytest.raises(DerivRestError, match="falhou"),
    ):
        asyncio.run(client._request("GET", "/trading/v1/options/accounts"))
    with (
        patch(
            "src.infrastructure.api.deriv_rest_client.read_http_response",
            return_value=json.dumps([]).encode(),
        ),
        pytest.raises(DerivRestError, match="JSON invalida"),
    ):
        asyncio.run(client._request("GET", "/trading/v1/options/accounts"))


def test_request_retries_transient_502_then_succeeds():
    client = DerivRestClient(
        rest_base_url="https://api.derivws.com",
        deriv_app_id="1089",
        access_token="t",
        timeout_seconds=5,
        max_retries=4,
    )
    err = urllib.error.HTTPError("u", 502, "Bad Gateway", {}, None)
    err.read = MagicMock(
        return_value=json.dumps(
            {
                "status": 502,
                "retryable": True,
                "retry_after": 60,
            }
        ).encode()
    )
    ok = json.dumps(
        {"data": [{"account_id": "DOT1", "balance": 1, "account_type": "demo", "status": "active"}]}
    ).encode()
    sleeps: list[float] = []

    def fake_sleep(seconds):
        sleeps.append(float(seconds))

    with (
        patch(
            "src.infrastructure.api.deriv_rest_client.read_http_response",
            side_effect=[err, ok],
        ),
        patch("src.infrastructure.api.deriv_rest_client.time.sleep", side_effect=fake_sleep),
    ):
        payload = client._request("GET", "/trading/v1/options/accounts")
    assert payload["data"][0]["account_id"] == "DOT1"
    assert sleeps == [60.0]


def test_request_exhausted_502_retries():
    client = DerivRestClient(
        rest_base_url="https://api.derivws.com",
        deriv_app_id="1089",
        access_token="t",
        timeout_seconds=5,
        max_retries=2,
    )
    err = urllib.error.HTTPError("u", 502, "Bad Gateway", {}, None)
    err.read = MagicMock(return_value=b'{"retry_after":2}')
    with (
        patch("src.infrastructure.api.deriv_rest_client.read_http_response", side_effect=err),
        patch("src.infrastructure.api.deriv_rest_client.time.sleep", return_value=None),
        pytest.raises(DerivRestError, match="HTTP 502"),
    ):
        client._request("GET", "/trading/v1/options/accounts")


@pytest.mark.asyncio
async def test_request_otp_ws_url_errors():
    client = DerivRestClient(
        rest_base_url="https://api.derivws.com",
        deriv_app_id="1089",
        access_token="t",
        timeout_seconds=5,
    )
    with (
        patch(
            "src.infrastructure.api.deriv_rest_client.read_http_response",
            return_value=json.dumps({"data": None}).encode(),
        ),
        pytest.raises(DerivRestError, match="otp"),
    ):
        await client.request_otp_ws_url("DOT1")
    with (
        patch(
            "src.infrastructure.api.deriv_rest_client.read_http_response",
            return_value=json.dumps({"data": {"url": ""}}).encode(),
        ),
        pytest.raises(DerivRestError, match="URL WebSocket"),
    ):
        await client.request_otp_ws_url("DOT1")


@pytest.mark.asyncio
async def test_request_with_json_body():
    client = DerivRestClient(
        rest_base_url="https://api.derivws.com",
        deriv_app_id="1089",
        access_token="t",
        timeout_seconds=5,
    )
    with patch(
        "src.infrastructure.api.deriv_rest_client.read_http_response",
        return_value=json.dumps({"data": {}}).encode(),
    ):
        out = await asyncio.to_thread(client._request, "POST", "/x", body={"a": 1})
    assert out == {"data": {}}


def test_retry_after_seconds_fallbacks():
    from src.infrastructure.api.deriv_rest_client import _retry_after_seconds

    assert _retry_after_seconds("not-json", attempt=0) == pytest.approx(2.0)
    assert _retry_after_seconds("[]", attempt=1) == pytest.approx(4.0)
    assert _retry_after_seconds('{"retry_after": null}', attempt=0) == pytest.approx(2.0)
    assert _retry_after_seconds('{"retry_after": "bad"}', attempt=2) == pytest.approx(6.0)
    assert _retry_after_seconds('{"retry_after": 120}', attempt=0) == pytest.approx(90.0)
