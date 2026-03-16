"""Shared utilities for livekit-wakeword."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


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
