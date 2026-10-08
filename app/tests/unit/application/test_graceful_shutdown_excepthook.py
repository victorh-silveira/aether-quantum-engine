import asyncio
import logging
import sys
from unittest.mock import patch

import run


def test_shutdown_safe_excepthook_ignores_closed_loop():
    old_hook = sys.excepthook
    try:
        run._original_excepthook = lambda *a: None
        run.install_shutdown_excepthook()
        assert sys.excepthook is run._shutdown_safe_excepthook
        sys.excepthook(RuntimeError, RuntimeError("Event loop is closed"), None)
        sys.excepthook(SystemExit, SystemExit(0), None)
    finally:
        sys.excepthook = old_hook


def test_shutdown_safe_excepthook_covers_branches():
    old_hook = sys.excepthook
    called: list[type[BaseException]] = []
    try:
        run._original_excepthook = lambda exc_type, exc_value, exc_tb: called.append(exc_type)
        run.install_shutdown_excepthook()
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
    old_hook = run._original_excepthook
    try:
        run._original_excepthook = lambda *args: (_ for _ in ()).throw(RuntimeError("hook down"))
        run._invoke_original_excepthook(ValueError, ValueError("boom"), None)
    finally:
        run._original_excepthook = old_hook


def test_shutdown_safe_excepthook_closed_loop():
    old_hook = sys.excepthook
    try:
        run._original_excepthook = lambda *a: None
        run.install_shutdown_excepthook()

        class _ClosedLoop:
            def is_closed(self):
                return True

        with patch("asyncio.get_running_loop", return_value=_ClosedLoop()):
            sys.excepthook(RuntimeError, RuntimeError("other"), None)
    finally:
        sys.excepthook = old_hook


def test_shutdown_safe_excepthook_ignores_cancelled_error_type():
    old_hook = sys.excepthook
    called: list[type[BaseException]] = []
    try:
        run._original_excepthook = lambda exc_type, exc_value, exc_tb: called.append(exc_type)
        run.install_shutdown_excepthook()

        sys.excepthook(asyncio.CancelledError, asyncio.CancelledError(), None)
        assert called == []
    finally:
        sys.excepthook = old_hook


def test_shutdown_safe_excepthook_ignores_none_type_message():
    old_hook = sys.excepthook
    called: list[type[BaseException]] = []
    try:
        run._original_excepthook = lambda exc_type, exc_value, exc_tb: called.append(exc_type)
        run.install_shutdown_excepthook()
        sys.excepthook(
            AttributeError,
            AttributeError("'NoneType' object has no attribute 'write'"),
            None,
        )
        assert called == []
    finally:
        sys.excepthook = old_hook


def test_shutdown_safe_excepthook_delegates_when_finalizing():
    old_hook = sys.excepthook
    called: list[type[BaseException]] = []
    try:
        run._original_excepthook = lambda exc_type, exc_value, exc_tb: called.append(exc_type)
        run.install_shutdown_excepthook()
        with patch.object(sys, "is_finalizing", return_value=True):
            sys.excepthook(ValueError, ValueError("late"), None)
        assert ValueError in called
    finally:
        sys.excepthook = old_hook


def test_shutdown_safe_excepthook_exits_cleanly_on_internal_failure():
    old_hook = sys.excepthook
    try:
        run._original_excepthook = lambda *a: None
        run.install_shutdown_excepthook()
        with (
            patch.object(run, "_should_delegate_to_native_hook", side_effect=RuntimeError("gc tore down")),
            patch.object(sys, "exit") as exit_mock,
        ):
            sys.excepthook(ValueError, ValueError("late"), None)
        exit_mock.assert_called_once_with(0)
    finally:
        sys.excepthook = old_hook


def test_logging_module_unavailable_when_get_logger_fails():
    with patch("run.logging.getLogger", side_effect=RuntimeError("logging gone")):
        assert run._logging_module_unavailable() is True


def test_logging_globally_desconfigured_when_manager_missing():
    with (
        patch.object(run, "_interpreter_finalizing", return_value=True),
        patch.object(logging.Logger, "manager", None),
    ):
        assert run._logging_globally_desconfigured() is True


def test_logging_globally_desconfigured_when_finalizing_without_handlers():
    with (
        patch.object(run, "_interpreter_finalizing", return_value=True),
        patch.object(logging.Logger, "manager", object()),
        patch.object(logging.root, "handlers", []),
    ):
        assert run._logging_globally_desconfigured() is True
