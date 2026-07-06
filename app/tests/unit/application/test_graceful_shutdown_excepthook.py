import sys
from unittest.mock import patch

import src.application.services.orchestrator.graceful_shutdown as graceful_shutdown_module
from src.application.services.orchestrator.graceful_shutdown import (
    _invoke_original_excepthook,
    _shutdown_safe_excepthook,
    install_shutdown_excepthook,
)


def test_shutdown_safe_excepthook_ignores_closed_loop():
    gs = graceful_shutdown_module

    old_hook = sys.excepthook
    try:
        gs._original_excepthook = lambda *a: None
        install_shutdown_excepthook()
        assert sys.excepthook is _shutdown_safe_excepthook
        sys.excepthook(RuntimeError, RuntimeError("Event loop is closed"), None)
        sys.excepthook(SystemExit, SystemExit(0), None)
    finally:
        sys.excepthook = old_hook


def test_shutdown_safe_excepthook_covers_branches():
    gs = graceful_shutdown_module

    old_hook = sys.excepthook
    called: list[type[BaseException]] = []
    try:
        gs._original_excepthook = lambda exc_type, exc_value, exc_tb: called.append(exc_type)
        install_shutdown_excepthook()
        sys.excepthook(GeneratorExit, GeneratorExit(), None)
        sys.excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)
        sys.excepthook(
            AttributeError,
            AttributeError("call_exception_handler failed"),
            None,
        )
        sys.excepthook(RuntimeError, RuntimeError("cannot schedule new futures after shutdown"), None)
        sys.excepthook(ValueError, ValueError("real"), None)
        assert ValueError in called
    finally:
        sys.excepthook = old_hook


def test_invoke_original_excepthook_swallows_hook_failure():
    gs = graceful_shutdown_module
    old_hook = gs._original_excepthook
    try:
        gs._original_excepthook = lambda *args: (_ for _ in ()).throw(RuntimeError("hook down"))
        _invoke_original_excepthook(ValueError, ValueError("boom"), None)
    finally:
        gs._original_excepthook = old_hook


def test_shutdown_safe_excepthook_closed_loop():
    gs = graceful_shutdown_module

    old_hook = sys.excepthook
    try:
        gs._original_excepthook = lambda *a: None
        install_shutdown_excepthook()

        class _ClosedLoop:
            def is_closed(self):
                return True

        with patch("asyncio.get_running_loop", return_value=_ClosedLoop()):
            sys.excepthook(RuntimeError, RuntimeError("other"), None)
    finally:
        sys.excepthook = old_hook
