"""End-to-end wake word detection pipeline."""

from __future__ import annotations

import tensorflow as tf

from ..config import WakeWordConfig
from ..models.feature_extractor import compute_n_frames
from .classifier import build_classifier


class SpecAugment(tf.keras.layers.Layer):  # type: ignore[misc]
    """SpecAugment: random time and frequency masking on spectrograms.

    During training, randomly zeroes out contiguous blocks of time frames
    and frequency channels to improve robustness. No-op during inference.

    Reference: Park et al., "SpecAugment", 2019.
    """

    def __init__(
        self,
        time_masks: int = 2,
        time_mask_max: int = 10,
        freq_masks: int = 2,
        freq_mask_max: int = 5,
        **kwargs: object,
    ):
        super().__init__(**kwargs)
        self.time_masks = time_masks
        self.time_mask_max = time_mask_max
        self.freq_masks = freq_masks
        self.freq_mask_max = freq_mask_max

    def call(self, x: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        if not training:
            return x

        shape = tf.shape(x)
        n_frames = shape[1]
        n_freq = shape[2]

        # Use index-based masking (XLA-compatible: no dynamic-shaped tensors)
        time_idx = tf.range(n_frames)  # (n_frames,)
        freq_idx = tf.range(n_freq)  # (n_freq,)

        mask = tf.ones_like(x)

        for _ in range(self.time_masks):
            t_len = tf.random.uniform([], 1, self.time_mask_max + 1, dtype=tf.int32)
            t_len = tf.minimum(t_len, n_frames)
            t_start = tf.random.uniform([], 0, n_frames - t_len + 1, dtype=tf.int32)
            # Boolean mask: True where index is outside [t_start, t_start + t_len)
            keep = (time_idx < t_start) | (time_idx >= t_start + t_len)
            # Reshape to (1, n_frames, 1) for broadcasting
            mask = mask * tf.cast(keep[None, :, None], x.dtype)

        for _ in range(self.freq_masks):
            f_len = tf.random.uniform([], 1, self.freq_mask_max + 1, dtype=tf.int32)
            f_len = tf.minimum(f_len, n_freq)
            f_start = tf.random.uniform([], 0, n_freq - f_len + 1, dtype=tf.int32)
            # Boolean mask: True where index is outside [f_start, f_start + f_len)
            keep = (freq_idx < f_start) | (freq_idx >= f_start + f_len)
            # Reshape to (1, 1, n_freq) for broadcasting
            mask = mask * tf.cast(keep[None, None, :], x.dtype)

        return x * mask

    def get_config(self) -> dict[str, object]:
        config = super().get_config()
        config.update(
            {
                "time_masks": self.time_masks,
                "time_mask_max": self.time_mask_max,
                "freq_masks": self.freq_masks,
                "freq_mask_max": self.freq_mask_max,
            }
        )
        return config


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
        self.spec_augment = SpecAugment()
        self.classifier = build_classifier(
            model_type=config.model.model_type,
            model_size=config.model.model_size,
            n_frames=n_frames,
            n_features=config.augmentation.num_channels,
        )

    def call(self, x: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        x = self.spec_augment(x, training=training)
        return self.classifier(x, training=training)
