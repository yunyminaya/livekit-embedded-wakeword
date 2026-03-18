"""Tests for inference pipeline components."""

from __future__ import annotations

import numpy as np

from livekit.embedded_wakeword.config import WakeWordConfig
from livekit.embedded_wakeword.models.pipeline import WakeWordClassifier


class TestWakeWordClassifier:
    def test_forward(self, sample_config: WakeWordConfig, sample_spectrogram: np.ndarray):
        model = WakeWordClassifier(sample_config)
        out = model(sample_spectrogram, training=False)
        assert out.shape == (4, 1)
        out_np = out.numpy()
        assert np.all(out_np >= 0) and np.all(out_np <= 1)

    def test_save_load(
        self, sample_config: WakeWordConfig, sample_spectrogram: np.ndarray, tmp_path
    ):
        model = WakeWordClassifier(sample_config)
        out1 = model(sample_spectrogram, training=False)

        # Save and reload weights
        path = tmp_path / "model.weights.h5"
        model.save_weights(str(path))

        model2 = WakeWordClassifier(sample_config)
        model2(sample_spectrogram, training=False)  # build first
        model2.load_weights(str(path))
        out2 = model2(sample_spectrogram, training=False)

        np.testing.assert_allclose(out1.numpy(), out2.numpy(), atol=1e-5)

    def test_spec_augment_masks_during_training(self):
        """SpecAugment should zero out some values during training but not inference."""
        import tensorflow as tf

        from livekit.embedded_wakeword.models.pipeline import SpecAugment

        # Large min masks guarantee zeroed regions regardless of randomness
        layer = SpecAugment(time_masks=1, time_mask_max=99, freq_masks=1, freq_mask_max=40)
        x = tf.ones((2, 99, 40))

        # Inference: no masking
        out_infer = layer(x, training=False)
        assert float(tf.reduce_min(out_infer)) == 1.0

        # Training: must have zeros — masks span the full axis
        out_train = layer(x, training=True)
        n_zeros = int(tf.reduce_sum(tf.cast(out_train == 0, tf.int32)))
        assert n_zeros > 0, "SpecAugment did not mask any values"
