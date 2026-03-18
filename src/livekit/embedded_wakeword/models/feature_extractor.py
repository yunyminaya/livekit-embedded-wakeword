"""Micro-speech spectrogram frontend for wake word detection.

Uses the TFLite Micro microfrontend to compute filterbank features matching
the microWakeWord training pipeline exactly.

Two backends are supported:

1. **TensorFlow** ``audio_microfrontend`` op (primary for training) — supports
   configurable ``window_step`` (20 ms default).
2. **pymicro-features** C extension (lightweight inference) — hardcoded to
   10 ms hop; frames are subsampled to match the configured step.

Parameters (matching microWakeWord / ESPHome micro_wake_word):
  - 16 kHz sample rate
  - 30 ms window
  - 20 ms hop (default, configurable)
  - 40 filterbank channels, 125–7500 Hz
  - PCAN gain control enabled
  - Log scale with shift=6

For a 2.0 s clip (32 000 samples) with 20 ms hop this produces 99 frames × 40 features.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Default micro_speech parameters
SAMPLE_RATE = 16000
WINDOW_SIZE_MS = 30
WINDOW_STEP_MS = 20
NUM_CHANNELS = 40
UPPER_BAND_LIMIT = 7500.0
LOWER_BAND_LIMIT = 125.0

# pymicro-features hardcoded step size
_PYMICRO_STEP_MS = 10

# Scaling factor to convert raw uint16 microfrontend output to float.
# The HuggingFace precomputed features store raw uint16 values and the
# dataset README instructs: "multiply by 0.0390625 after casting to float".
# pymicro-features already applies this scaling internally.
_UINT16_TO_FLOAT_SCALE = 0.0390625  # = 1/25.6


def compute_n_frames(
    total_samples: int,
    window_size_ms: int = WINDOW_SIZE_MS,
    window_step_ms: int = WINDOW_STEP_MS,
    sample_rate: int = SAMPLE_RATE,
) -> int:
    """Compute number of spectrogram frames for a given audio length."""
    window_samples = window_size_ms * sample_rate // 1000
    hop_samples = window_step_ms * sample_rate // 1000
    return max(0, (total_samples - window_samples) // hop_samples + 1)


def _generate_features_tf(
    audio_int16: np.ndarray,
    step_ms: int = WINDOW_STEP_MS,
) -> np.ndarray:
    """Generate features using TensorFlow's audio_microfrontend op.

    This is the same algorithm as pymicro-features but implemented as a TF op,
    with configurable step size.

    Args:
        audio_int16: int16 audio samples at 16 kHz.
        step_ms: Hop size in milliseconds (default 20).

    Returns:
        (N_frames, 40) float32 spectrogram features.
    """
    import tensorflow as tf
    from tensorflow.lite.experimental.microfrontend.python.ops import (  # type: ignore[import-untyped]
        audio_microfrontend as frontend_op,
    )

    with tf.device("/cpu:0"):
        result = frontend_op.audio_microfrontend(
            tf.convert_to_tensor(audio_int16),
            sample_rate=SAMPLE_RATE,
            window_size=WINDOW_SIZE_MS,
            window_step=step_ms,
            num_channels=NUM_CHANNELS,
            upper_band_limit=UPPER_BAND_LIMIT,
            lower_band_limit=LOWER_BAND_LIMIT,
            enable_pcan=True,
            min_signal_remaining=0.05,
            out_scale=1,
            out_type=tf.uint16,
        )
        # Scale uint16 → float32 to match pymicro-features output convention
        return result.numpy().astype(np.float32) * _UINT16_TO_FLOAT_SCALE


def _generate_features_pymicro(
    audio_int16: np.ndarray,
    step_ms: int = WINDOW_STEP_MS,
) -> np.ndarray:
    """Generate features using pymicro-features (C extension).

    The pymicro-features package wraps the TFLite Micro microfrontend C library.
    It is hardcoded to 10 ms hop.  When step_ms > 10, frames are subsampled
    to approximate the requested step.

    Args:
        audio_int16: int16 audio samples at 16 kHz.
        step_ms: Desired hop size in milliseconds (default 20).

    Returns:
        (N_frames, 40) float32 spectrogram features.
    """
    from pymicro_features import MicroFrontend as _PMF

    audio_bytes = audio_int16.tobytes()
    frontend = _PMF()

    features: list[list[float]] = []
    audio_idx = 0
    chunk_bytes = 160 * 2  # 10ms of int16 audio = 160 samples * 2 bytes
    num_bytes = len(audio_bytes)

    while audio_idx + chunk_bytes <= num_bytes:
        result = frontend.process_samples(audio_bytes[audio_idx : audio_idx + chunk_bytes])
        audio_idx += result.samples_read * 2
        if result.features:
            features.append(result.features)

    if not features:
        return np.zeros((0, NUM_CHANNELS), dtype=np.float32)

    arr = np.array(features, dtype=np.float32)
    # pymicro-features already outputs scaled floats (uint16 * 0.0390625, range ~0-26).
    # No additional scaling needed.

    # pymicro-features always uses 10ms hop. Subsample if a larger step is needed.
    if step_ms > _PYMICRO_STEP_MS:
        subsample_factor = step_ms // _PYMICRO_STEP_MS
        arr = arr[::subsample_factor]

    return arr


class MicroFrontend:
    """Micro-speech spectrogram frontend using the TFLite Micro microfrontend.

    Uses TensorFlow's ``audio_microfrontend`` op as the primary backend
    (supports configurable step size), falling back to ``pymicro-features``
    (C extension, subsampled from 10 ms to the configured step).

    Both backends produce the same filterbank features matching the on-device
    ESPHome implementation and microWakeWord's precomputed training features.

    Output: (batch, N_frames, 40) float32 spectrogram features
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        window_size_ms: int = WINDOW_SIZE_MS,
        window_step_ms: int = WINDOW_STEP_MS,
        num_channels: int = NUM_CHANNELS,
    ):
        self.sample_rate = sample_rate
        self.window_size_ms = window_size_ms
        self.window_step_ms = window_step_ms
        self.num_channels = num_channels

        # Detect available backend
        self._backend = self._detect_backend()
        logger.info(
            "MicroFrontend: %d ms window, %d ms hop, %d channels, backend=%s",
            window_size_ms,
            window_step_ms,
            num_channels,
            self._backend,
        )

    @staticmethod
    def _detect_backend() -> str:
        """Detect best available feature generation backend.

        Prefers TensorFlow when installed (native step_ms support), falls
        back to pymicro-features (core dependency, always available).
        """
        try:
            from tensorflow.lite.experimental.microfrontend.python.ops import (  # type: ignore[import-untyped]
                audio_microfrontend as _op,  # noqa: F401
            )

            return "tensorflow"
        except (ImportError, ModuleNotFoundError):
            pass
        # pymicro-features is a core dependency — should always be available
        from pymicro_features import MicroFrontend as _MF  # noqa: F401

        return "pymicro-features"

    def _generate_features(self, audio_int16: np.ndarray) -> np.ndarray:
        """Generate features for a single audio clip."""
        if self._backend == "tensorflow":
            return _generate_features_tf(audio_int16, self.window_step_ms)
        else:
            return _generate_features_pymicro(audio_int16, self.window_step_ms)

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """Compute spectrogram features.

        Args:
            audio: Audio at 16 kHz. Can be float32 [-1, 1] or int16.
                Shape (samples,) or (batch, samples).

        Returns:
            (batch, N_frames, num_channels) float32 spectrogram features.
        """
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]

        # Convert float32 to int16 (matching microWakeWord convention)
        if audio.dtype in (np.float32, np.float64):
            audio = np.clip(audio * 32768, -32768, 32767).astype(np.int16)

        batch_size = audio.shape[0]
        results = []

        for b in range(batch_size):
            features = self._generate_features(audio[b])
            results.append(features)

        # Pad to uniform frame count within the batch
        if results:
            max_frames = max(f.shape[0] for f in results)
            padded = []
            for f in results:
                if f.shape[0] < max_frames:
                    pad = np.zeros((max_frames - f.shape[0], self.num_channels), dtype=np.float32)
                    f = np.concatenate([pad, f], axis=0)
                padded.append(f)
            return np.stack(padded, axis=0)

        return np.zeros((batch_size, 0, self.num_channels), dtype=np.float32)

    @property
    def n_frames_for_duration(self) -> dict[float, int]:
        """Common clip durations → frame counts."""
        return {
            dur: compute_n_frames(
                int(dur * self.sample_rate),
                self.window_size_ms,
                self.window_step_ms,
                self.sample_rate,
            )
            for dur in [1.0, 1.5, 2.0, 2.5, 3.0]
        }
