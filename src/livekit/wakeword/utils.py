"""Shared utilities for livekit-wakeword."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Return the best available torch device (CUDA > MPS > CPU)."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_gpu() -> None:
    """Configure TensorFlow GPU memory growth if available."""
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            logger.info(f"TensorFlow using {len(gpus)} GPU(s)")
        else:
            logger.info("TensorFlow using CPU")
    except Exception as e:
        logger.warning(f"GPU configuration failed: {e}")
