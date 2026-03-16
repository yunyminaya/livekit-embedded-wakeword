"""Tests for Inception and MixedNet classifiers."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from livekit.wakeword.config import ModelSize, ModelType
from livekit.wakeword.models.classifier import (
    InceptionClassifier,
    MixedNetClassifier,
    build_classifier,
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


class TestBuildClassifier:
    def test_build_inception(self, sample_spectrogram: np.ndarray):
        model = build_classifier(ModelType.inception, ModelSize.small)
        out = model(sample_spectrogram, training=False)
        assert out.shape == (4, 1)

    def test_build_mixednet(self, sample_spectrogram: np.ndarray):
        model = build_classifier(ModelType.mixednet, ModelSize.small)
        out = model(sample_spectrogram, training=False)
        assert out.shape == (4, 1)

    def test_all_sizes(self, sample_spectrogram: np.ndarray):
        for size in ModelSize:
            for mtype in ModelType:
                model = build_classifier(mtype, size)
                out = model(sample_spectrogram, training=False)
                assert out.shape == (4, 1)
