"""Ponto de entrada: carrega configuracao e executa o orquestrador."""

import asyncio
import logging
import sys
import types

from aether_paths import REPO_ROOT
from src.application.services.orchestrator import Orchestrator
from src.application.services.orchestrator.engine_mode import ENGINE_MODE_EXECUTE
from src.application.services.orchestrator.engine_session import (
    create_authenticated_auth,
    load_engine_config,
)


_SHUTDOWN_EXC_MARKERS = (
    "Event loop is closed",
    "CancelledError",
    "NoneType object",
    "cannot schedule new futures",
    "call_exception_handler",
)

_original_excepthook = sys.excepthook


def _interpreter_finalizing() -> bool:
    """Indica se o interpretador Python esta no estagio final de encerramento."""
    if sys is None:
        return True
    is_finalizing = getattr(sys, "is_finalizing", None)
    if callable(is_finalizing):
        return bool(is_finalizing())
    return False


def _logging_module_unavailable() -> bool:
    """Indica se o modulo logging foi desmontado ou esta inacessivel."""
    if logging is None:
        return True
    try:
        root = logging.getLogger()
    except Exception:
        return True
    return root is None


def _logging_globally_desconfigured() -> bool:
    """Indica se loggers globais perderam handlers durante finalizacao do interpretador."""
    if not _interpreter_finalizing():
        return False
    try:
        manager = getattr(logging.Logger, "manager", None)
        if manager is None:
            return True
        return not logging.root.handlers
    except Exception:
        return True


def _should_delegate_to_native_hook() -> bool:
    """Indica se o excepthook nativo deve assumir sem log estruturado da aplicacao."""
    return (
        _logging_module_unavailable() or _logging_globally_desconfigured() or sys is None or _interpreter_finalizing()
    )


def _exception_is_shutdown_noise(
    exc_type: type[BaseException],
    exc_value: BaseException,
) -> bool:
    """Indica excecao benigna de cancelamento ou loop fechado no desmonte assincrono."""
    if exc_type in (SystemExit, GeneratorExit, KeyboardInterrupt):
        return True
    if exc_type is asyncio.CancelledError:
        return True
    message = str(exc_value)
    if any(marker in message for marker in _SHUTDOWN_EXC_MARKERS):
        return True
    if exc_type is RuntimeError and ("Event loop is closed" in message or "cannot schedule new futures" in message):
        return True
    return exc_type is AttributeError and ("call_exception_handler" in message or "NoneType" in message)


def _invoke_original_excepthook(
    exc_type: type[BaseException],
    exc_value: BaseException,
    exc_tb: types.TracebackType | None,
) -> None:
    """Delega ao excepthook nativo sem propagar falhas do proprio hook."""
    try:
        _original_excepthook(exc_type, exc_value, exc_tb)
    except BaseException:
        return


def _shutdown_safe_excepthook(
    exc_type: type[BaseException],
    exc_value: BaseException,
    exc_tb: types.TracebackType | None,
) -> None:
    """Barreira defensiva contra Error in sys.excepthook no desmonte Windows."""
    try:
        if _should_delegate_to_native_hook():
            _invoke_original_excepthook(exc_type, exc_value, exc_tb)
            return
        if _exception_is_shutdown_noise(exc_type, exc_value):
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None and loop.is_closed():
            return
        _invoke_original_excepthook(exc_type, exc_value, exc_tb)
    except BaseException:
        try:
            sys.exit(0)
        except BaseException:
            return


def install_shutdown_excepthook() -> None:
    """Instala excepthook que neutraliza ruido de GC no desmonte assincrono Windows."""
    sys.excepthook = _shutdown_safe_excepthook


def _emit_fatal_startup_error(exc: BaseException) -> None:
    """Emite erro fatal no stderr quando streams ainda estao abertos."""
    try:
        stderr = getattr(sys, "stderr", None)
        if stderr is None or getattr(stderr, "closed", False):
            return
        print(f"ERRO fatal ao iniciar motor: {exc}", file=stderr, flush=True)
    except Exception:
        return


async def main() -> int:
    """Carrega configuracao, autentica e executa o loop principal do motor."""
    config, logger = load_engine_config(engine_mode=ENGINE_MODE_EXECUTE)
    auth = create_authenticated_auth(config, logger)
    if auth is None:
        return 1

    orchestrator = Orchestrator(config, auth)
    try:
        await orchestrator.run()
    except (asyncio.CancelledError, KeyboardInterrupt):
        return 130
    finally:
        await orchestrator.close_infrastructure_connections()

    reason = getattr(orchestrator, "shutdown_reason", None)
    if reason == "stop_win":
        target = orchestrator.risk_manager.total_session_profit
        logger.info("STOP_WIN: meta da sessao atingida (pnl_sessao=$%+.2f). Motor encerrado.", target)
        return 0
    if not orchestrator.running:
        logger.error(
            "Motor encerrou antes do loop principal. Veja INIT (PAT, OTP, stream) e %s",
            REPO_ROOT / ".env",
        )
        return 1
    return 0


if __name__ == "__main__":
    install_shutdown_excepthook()
    try:
        sys.exit(asyncio.run(main()))
    except (SystemExit, KeyboardInterrupt):
        raise
    except Exception as exc:
        _emit_fatal_startup_error(exc)
        sys.exit(1)
