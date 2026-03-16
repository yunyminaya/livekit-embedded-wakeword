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
    """Configure TensorFlow GPU/MPS memory growth if available.

    Must be called before any TensorFlow operations to take effect.
    Works with CUDA (Linux/Windows) and Metal/MPS (macOS via tensorflow-metal).
    """
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                # Memory growth must be set before GPUs are initialized
                pass
        if gpus:
            names = [g.name for g in gpus]
            logger.info(f"TensorFlow using {len(gpus)} GPU(s): {names}")
        else:
            logger.info("TensorFlow: no GPU found, using CPU")
    except Exception as e:
        logger.warning(f"GPU configuration failed: {e}")
