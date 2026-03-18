"""Tests for Inception, MixedNet, RepCNN, and WakeNet9 classifiers."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from livekit.embedded_wakeword.config import ModelSize, ModelType
from livekit.embedded_wakeword.models.classifier import (
    InceptionClassifier,
    MixedNetClassifier,
    RepCNNClassifier,
    WakeNet9Classifier,
    build_classifier,
    reparameterize_model,
)


class TestInceptionClassifier:
    def test_output_shape(self, sample_spectrogram: np.ndarray):
        model = InceptionClassifier(n_frames=99, n_features=40, filters=8, n_inception_blocks=2)
        out = model(sample_spectrogram, training=False)
        assert out.shape == (4, 1)

    def test_output_range(self, sample_spectrogram: np.ndarray):
        model = InceptionClassifier(n_frames=99, n_features=40, filters=8, n_inception_blocks=2)
        out = model(sample_spectrogram, training=False).numpy()
        assert np.all(out >= 0) and np.all(out <= 1)

    def test_training_mode(self, sample_spectrogram: np.ndarray):
        model = InceptionClassifier(n_frames=99, n_features=40, filters=8, n_inception_blocks=2)
        out = model(sample_spectrogram, training=True)
        assert out.shape == (4, 1)


class TestMixedNetClassifier:
    def test_output_shape(self, sample_spectrogram: np.ndarray):
        model = MixedNetClassifier(n_frames=99, n_features=40, filters=16, n_blocks=2)
        out = model(sample_spectrogram, training=False)
        assert out.shape == (4, 1)

    def test_output_range(self, sample_spectrogram: np.ndarray):
        model = MixedNetClassifier(n_frames=99, n_features=40, filters=16, n_blocks=2)
        out = model(sample_spectrogram, training=False).numpy()
        assert np.all(out >= 0) and np.all(out <= 1)


class TestRepCNNClassifier:
    def test_output_shape(self, sample_spectrogram: np.ndarray):
        model = RepCNNClassifier(n_frames=99, n_features=40, filters=8, n_blocks=2)
        out = model(sample_spectrogram, training=False)
        assert out.shape == (4, 1)

    def test_output_range(self, sample_spectrogram: np.ndarray):
        model = RepCNNClassifier(n_frames=99, n_features=40, filters=8, n_blocks=2)
        out = model(sample_spectrogram, training=False).numpy()
        assert np.all(out >= 0) and np.all(out <= 1)

    def test_training_mode(self, sample_spectrogram: np.ndarray):
        model = RepCNNClassifier(n_frames=99, n_features=40, filters=8, n_blocks=2)
        out = model(sample_spectrogram, training=True)
        assert out.shape == (4, 1)


class TestWakeNet9Classifier:
    def test_output_shape(self, sample_spectrogram: np.ndarray):
        model = WakeNet9Classifier(n_frames=99, n_features=40, filters=16, n_blocks=3)
        out = model(sample_spectrogram, training=False)
        assert out.shape == (4, 1)

    def test_output_range(self, sample_spectrogram: np.ndarray):
        model = WakeNet9Classifier(n_frames=99, n_features=40, filters=16, n_blocks=3)
        out = model(sample_spectrogram, training=False).numpy()
        assert np.all(out >= 0) and np.all(out <= 1)

    def test_training_mode(self, sample_spectrogram: np.ndarray):
        model = WakeNet9Classifier(n_frames=99, n_features=40, filters=16, n_blocks=3)
        out = model(sample_spectrogram, training=True)
        assert out.shape == (4, 1)

    def test_dilation_cycling(self):
        model = WakeNet9Classifier(n_frames=99, n_features=40, filters=8, n_blocks=7)
        dilations = [b._dilation_rate for b in model.liglu_blocks]
        assert dilations == [1, 2, 3, 4, 3, 1, 2]

    def test_long_range_residual_disabled_small(self, sample_spectrogram: np.ndarray):
        model = WakeNet9Classifier(n_frames=99, n_features=40, filters=8, n_blocks=2)
        assert not model._use_long_residual
        out = model(sample_spectrogram, training=False)
        assert out.shape == (4, 1)


class TestRepDSBlockFusion:
    def test_fuse_matches_forward(self, sample_spectrogram: np.ndarray):
        """Critical: fused model must produce identical output to multi-branch model."""
        model = RepCNNClassifier(n_frames=99, n_features=40, filters=8, n_blocks=3)
        # Build the model
        model(sample_spectrogram, training=False)
        # Run a few training steps to get non-trivial BN statistics
        for _ in range(5):
            model(sample_spectrogram, training=True)

        # Get multi-branch output (inference mode)
        original_out = model(sample_spectrogram, training=False).numpy()

        # Fuse and get output
        fused = reparameterize_model(model)
        fused_out = fused(sample_spectrogram, training=False).numpy()

        np.testing.assert_allclose(original_out, fused_out, atol=1e-4)

    def test_fused_model_has_no_batchnorm(self, sample_spectrogram: np.ndarray):
        model = RepCNNClassifier(n_frames=99, n_features=40, filters=8, n_blocks=2)
        model(sample_spectrogram, training=False)
        fused = reparameterize_model(model)
        for layer in fused.layers:
            assert not isinstance(layer, tf.keras.layers.BatchNormalization)

    def test_dilation_cycling(self):
        model = RepCNNClassifier(n_frames=99, n_features=40, filters=8, n_blocks=6)
        dilations = [b._dilation for b in model.rep_blocks]
        assert dilations == [1, 2, 4, 1, 2, 4]

    def test_fused_has_depthwise_and_pointwise(self, sample_spectrogram: np.ndarray):
        """Fused model should have DepthwiseConv2D + Conv2D pairs, not plain Conv2D."""
        model = RepCNNClassifier(n_frames=99, n_features=40, filters=8, n_blocks=2)
        model(sample_spectrogram, training=False)
        fused = reparameterize_model(model)
        dw_layers = [ly for ly in fused.layers if isinstance(ly, tf.keras.layers.DepthwiseConv2D)]
        assert len(dw_layers) == 2  # one per block


class TestBuildClassifier:
    def test_build_inception(self, sample_spectrogram: np.ndarray):
        model = build_classifier(ModelType.inception, ModelSize.small)
        out = model(sample_spectrogram, training=False)
        assert out.shape == (4, 1)

    def test_build_mixednet(self, sample_spectrogram: np.ndarray):
        model = build_classifier(ModelType.mixednet, ModelSize.small)
        out = model(sample_spectrogram, training=False)
        assert out.shape == (4, 1)

    def test_build_repcnn(self, sample_spectrogram: np.ndarray):
        model = build_classifier(ModelType.repcnn, ModelSize.small)
        out = model(sample_spectrogram, training=False)
        assert out.shape == (4, 1)

    def test_build_wakenet9(self, sample_spectrogram: np.ndarray):
        model = build_classifier(ModelType.wakenet9, ModelSize.small)
        out = model(sample_spectrogram, training=False)
        assert out.shape == (4, 1)

    def test_all_sizes(self, sample_spectrogram: np.ndarray):
        for size in ModelSize:
            for mtype in ModelType:
                model = build_classifier(mtype, size)
                out = model(sample_spectrogram, training=False)
                assert out.shape == (4, 1)
