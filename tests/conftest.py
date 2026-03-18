"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from livekit.embedded_wakeword.config import WakeWordConfig


@pytest.fixture
def sample_config(tmp_path: Path) -> WakeWordConfig:
    """Minimal config for testing."""
    return WakeWordConfig(
        model_name="test_wakeword",
        target_phrases=["hey test"],
        n_samples=10,
        n_samples_val=5,
        data_dir=str(tmp_path / "data"),
        output_dir=str(tmp_path / "output"),
    )


@pytest.fixture
def sample_audio() -> np.ndarray:
    """Random 2-second 16kHz audio batch."""
    return np.random.randn(2, 32000).astype(np.float32)


@pytest.fixture
def sample_spectrogram() -> np.ndarray:
    """Random (batch, 99, 40) spectrogram features."""
    return np.random.randn(4, 99, 40).astype(np.float32)


@pytest.fixture
def sample_features_file(tmp_path: Path) -> Path:
    """Create a temporary .npy features file."""
    features = np.random.randn(100, 99, 40).astype(np.float32)
    path = tmp_path / "test_features.npy"
    np.save(str(path), features)
    return path
