"""End-to-end wake word detection pipeline."""

from __future__ import annotations

import tensorflow as tf

from ..config import WakeWordConfig
from ..models.feature_extractor import compute_n_frames
from .classifier import build_classifier


class WakeWordClassifier(tf.keras.Model):  # type: ignore[misc]
    """Classifier-only module operating on pre-extracted spectrogram features.

    Used during training when features are pre-computed.
    Input: (batch, N_frames, 40) → Output: (batch, 1)
    """

    def __init__(self, config: WakeWordConfig):
        super().__init__()
        n_frames = compute_n_frames(
            int(config.augmentation.clip_duration * 16000),
            config.augmentation.window_size_ms,
            config.augmentation.window_step_ms,
        )
        self.classifier = build_classifier(
            model_type=config.model.model_type,
            model_size=config.model.model_size,
            n_frames=n_frames,
            n_features=config.augmentation.num_channels,
        )

    def call(
        self, x: tf.Tensor, training: bool | None = None
    ) -> tf.Tensor:
        return self.classifier(x, training=training)
