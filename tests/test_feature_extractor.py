"""Tests for MicroFrontend spectrogram extractor."""

from __future__ import annotations

import numpy as np

from livekit.embedded_wakeword.models.feature_extractor import MicroFrontend, compute_n_frames


class TestMicroFrontend:
    """Test the TFLite Micro microfrontend spectrogram extractor."""

    def test_output_shape(self):
        frontend = MicroFrontend()
        audio = np.random.randn(2, 32000).astype(np.float32)
        spec = frontend(audio)
        batch, n_frames, n_channels = spec.shape
        assert batch == 2
        assert n_channels == 40
        assert n_frames == 99

    def test_single_sample(self):
        frontend = MicroFrontend()
        audio = np.random.randn(1, 16000).astype(np.float32)  # 1 second
        spec = frontend(audio)
        assert spec.shape[0] == 1
        assert spec.shape[2] == 40
        assert spec.shape[1] > 0

    def test_1d_input(self):
        """1D input should be auto-batched."""
        frontend = MicroFrontend()
        audio = np.random.randn(16000).astype(np.float32)
        spec = frontend(audio)
        assert spec.ndim == 3
        assert spec.shape[0] == 1
        assert spec.shape[2] == 40

    def test_deterministic(self):
        frontend = MicroFrontend()
        audio = np.random.randn(1, 16000).astype(np.float32)
        spec1 = frontend(audio)
        spec2 = frontend(audio)
        np.testing.assert_allclose(spec1, spec2)

    def test_custom_step(self):
        """Custom hop size should produce more frames."""
        frontend_20 = MicroFrontend(window_step_ms=20)
        frontend_10 = MicroFrontend(window_step_ms=10)
        audio = np.random.randn(16000).astype(np.float32)
        spec_20 = frontend_20(audio)
        spec_10 = frontend_10(audio)
        assert spec_10.shape[1] > spec_20.shape[1]
        assert spec_20.shape[2] == 40
        assert spec_10.shape[2] == 40

    def test_short_audio(self):
        """Very short audio should produce 0 frames."""
        frontend = MicroFrontend()
        audio = np.random.randn(100).astype(np.float32)  # < 30ms
        spec = frontend(audio)
        assert spec.shape[1] == 0


class TestComputeNFrames:
    def test_2s_audio(self):
        """2.0s at 16kHz with 30ms/20ms should produce 99 frames."""
        assert compute_n_frames(32000) == 99

    def test_1s_audio(self):
        n = compute_n_frames(16000)
        assert n > 0
        assert n < 99

    def test_zero_samples(self):
        assert compute_n_frames(0) == 0
