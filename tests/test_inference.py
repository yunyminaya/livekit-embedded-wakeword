"""Tests for inference pipeline components."""

from __future__ import annotations

import numpy as np

from livekit.wakeword.config import WakeWordConfig
from livekit.wakeword.models.pipeline import WakeWordClassifier


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
