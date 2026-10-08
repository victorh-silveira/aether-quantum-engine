import urllib.request
from unittest.mock import MagicMock, patch

import pytest

from src.infrastructure.api.deriv_http import read_http_response


def test_read_http_response_rejects_unsafe_scheme():
    req = urllib.request.Request("file:///etc/passwd", method="GET")
    with pytest.raises(ValueError, match="Esquema"):
        read_http_response(req, 5.0)


def test_read_http_response_returns_body():
    payload = b'{"ok":true}'
    resp = MagicMock()
    resp.read.return_value = payload
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    req = urllib.request.Request("https://api.derivws.com/v1/health", method="GET")
    with patch("urllib.request.urlopen", return_value=resp):
        assert read_http_response(req, 5.0) == payload
