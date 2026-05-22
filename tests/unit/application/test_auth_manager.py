import os
from unittest.mock import patch

from src.application.services.auth_manager import AuthManager


def test_auth_manager_demo_token_from_env():
    with patch.dict(os.environ, {"AETHER_DEMO_TOKEN": "demo-secret"}, clear=False):
        mgr = AuthManager(mode="demo")
    assert mgr.get_token() == "demo-secret"


def test_auth_manager_live_mode_uses_live_suffix():
    with patch.dict(os.environ, {"AETHER_LIVE_TOKEN": "live-secret"}, clear=False):
        mgr = AuthManager(mode="live")
    assert mgr.get_token() == "live-secret"
