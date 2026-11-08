import threading

from src.application.services.deep_learning.dl_symbol_runtime import guard_symbol_model


def test_guard_symbol_model_without_lock():
    with guard_symbol_model({}):
        pass


def test_guard_symbol_model_reentrant_same_thread():
    runtime = {"model_lock": threading.RLock()}
    with guard_symbol_model(runtime), guard_symbol_model(runtime):
        pass


def test_guard_symbol_model_serializes_access():
    lock = threading.RLock()
    runtime = {"model_lock": lock}
    order = []

    def worker(tag: str):
        with guard_symbol_model(runtime):
            order.append(f"{tag}-in")
            order.append(f"{tag}-out")

    threads = [threading.Thread(target=worker, args=(str(i),)) for i in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for idx in range(0, len(order), 2):
        assert order[idx].endswith("-in")
        assert order[idx + 1].endswith("-out")
