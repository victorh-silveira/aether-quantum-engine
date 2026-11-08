"""Utilitario opcional de limpeza de cache CUDA no motor de execucao."""

from __future__ import annotations


def clear_cuda_cache() -> None:
    """Libera cache CUDA quando o runtime Torch estiver disponivel."""
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
