"""Tests for the eval module."""

from __future__ import annotations

import numpy as np

from livekit.wakeword.eval.evaluate import (
    _compute_aut,
    _compute_det_curve,
    _predict_tflite,
)


class TestDetCurve:
    """Tests for DET curve computation."""

    def test_perfect_separation(self) -> None:
        """Perfect model: positives all 1.0, negatives all 0.0."""
        pos = np.ones(100)
        neg = np.zeros(100)
        thresholds, fpr, fnr = _compute_det_curve(pos, neg)

        # At threshold 0.5: FPR=0 (no negatives >= 0.5), FNR=0 (all positives >= 0.5)
        idx_05 = np.argmin(np.abs(thresholds - 0.5))
        assert fpr[idx_05] == 0.0
        assert fnr[idx_05] == 0.0

    def test_random_model(self) -> None:
        """Random scores should have non-trivial FPR and FNR."""
        rng = np.random.RandomState(42)
        pos = rng.rand(500)
        neg = rng.rand(500)
        thresholds, fpr, fnr = _compute_det_curve(pos, neg)

        # At threshold 0.5: roughly half should be misclassified
        idx_05 = np.argmin(np.abs(thresholds - 0.5))
        assert 0.3 < fpr[idx_05] < 0.7
        assert 0.3 < fnr[idx_05] < 0.7

    def test_shape(self) -> None:
        """Output arrays should all be the same length."""
        pos = np.random.rand(50)
        neg = np.random.rand(50)
        thresholds, fpr, fnr = _compute_det_curve(pos, neg)
        assert thresholds.shape == fpr.shape == fnr.shape


class TestAUT:
    """Tests for Area Under the DET curve."""

    def test_perfect_model_aut_near_zero(self) -> None:
        """Perfect separation should yield AUT near 0."""
        pos = np.ones(100)
        neg = np.zeros(100)
        _, fpr, fnr = _compute_det_curve(pos, neg)
        aut = _compute_aut(fpr, fnr)
        assert aut < 0.01

    def test_random_model_aut(self) -> None:
        """Random model should have AUT significantly above 0."""
        rng = np.random.RandomState(42)
        pos = rng.rand(500)
        neg = rng.rand(500)
        _, fpr, fnr = _compute_det_curve(pos, neg)
        aut = _compute_aut(fpr, fnr)
        assert 0.1 < aut < 0.5

    def test_aut_bounded(self) -> None:
        """AUT should be between 0 and 1."""
        pos = np.random.rand(100)
        neg = np.random.rand(100)
        _, fpr, fnr = _compute_det_curve(pos, neg)
        aut = _compute_aut(fpr, fnr)
        assert 0.0 <= aut <= 1.0


class TestTflitePredict:
    """Tests for TFLite prediction helper."""

    def test_predict_with_mock_interpreter(self) -> None:
        """Ensure _predict_tflite processes and concatenates correctly."""

        class _MockInterpreter:
            def __init__(self):
                self._input = None

            def get_input_details(self):
                return [{"index": 0, "shape": [1, 99, 40]}]

            def get_output_details(self):
                return [{"index": 1}]

            def set_tensor(self, index, data):
                self._input = data

            def invoke(self):
                pass

            def get_tensor(self, index):
                # Return mean of input as score
                batch = self._input
                score = batch.mean(axis=(1, 2))[:, np.newaxis]
                return score

            def allocate_tensors(self):
                pass

        features = np.random.randn(50, 99, 40).astype(np.float32)
        scores = _predict_tflite(_MockInterpreter(), features, batch_size=16)
        assert scores.shape == (50,)
