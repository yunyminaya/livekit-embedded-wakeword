"""Micro-speech spectrogram frontend for wake word detection.

Computes a mel-frequency spectrogram matching the micro_speech TFLite
preprocessor parameters used by microWakeWord:
  - 16 kHz sample rate
  - 30 ms window (480 samples), 20 ms hop (320 samples)
  - 40 mel-frequency channels, 125–7500 Hz
  - Log compression

For a 2.0 s clip (32 000 samples) this produces 99 frames × 40 features.
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


def _hz_to_mel(hz: float) -> float:
    """Convert frequency in Hz to mel scale."""
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    """Convert mel scale to frequency in Hz."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _build_mel_filterbank(
    num_fft_bins: int,
    num_channels: int = NUM_CHANNELS,
    sample_rate: int = SAMPLE_RATE,
    lower_hz: float = LOWER_BAND_LIMIT,
    upper_hz: float = UPPER_BAND_LIMIT,
) -> np.ndarray:
    """Build a mel-spaced triangular filterbank matrix.

    Returns:
        (num_channels, num_fft_bins) filterbank weights.
    """
    low_mel = _hz_to_mel(lower_hz)
    high_mel = _hz_to_mel(upper_hz)

    # num_channels + 2 points to define num_channels triangular filters
    mel_points = np.linspace(low_mel, high_mel, num_channels + 2)
    hz_points = np.array([_mel_to_hz(m) for m in mel_points])
    bin_points = np.floor((num_fft_bins * 2) * hz_points / sample_rate).astype(int)

    filterbank = np.zeros((num_channels, num_fft_bins), dtype=np.float32)
    for i in range(num_channels):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]

        for j in range(left, center):
            if center > left:
                filterbank[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right > center:
                filterbank[i, j] = (right - j) / (right - center)

    return filterbank


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


class MicroFrontend:
    """Micro-speech spectrogram frontend (pure numpy).

    Computes log-mel spectrogram features matching the micro_speech
    TFLite preprocessing model parameters.

    Output: (batch, N_frames, 40) float32 log-mel features
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        window_size_ms: int = WINDOW_SIZE_MS,
        window_step_ms: int = WINDOW_STEP_MS,
        num_channels: int = NUM_CHANNELS,
        upper_band_limit: float = UPPER_BAND_LIMIT,
        lower_band_limit: float = LOWER_BAND_LIMIT,
    ):
        self.sample_rate = sample_rate
        self.window_size_ms = window_size_ms
        self.window_step_ms = window_step_ms
        self.num_channels = num_channels

        self._window_samples = window_size_ms * sample_rate // 1000
        self._hop_samples = window_step_ms * sample_rate // 1000
        self._fft_size = 1 << (self._window_samples - 1).bit_length()  # next power of 2
        self._num_fft_bins = self._fft_size // 2 + 1

        # Hann window
        self._window = np.hanning(self._window_samples).astype(np.float32)

        # Mel filterbank: (num_channels, num_fft_bins)
        self._filterbank = _build_mel_filterbank(
            num_fft_bins=self._num_fft_bins,
            num_channels=num_channels,
            sample_rate=sample_rate,
            lower_hz=lower_band_limit,
            upper_hz=upper_band_limit,
        )

        logger.info(
            "MicroFrontend: %d ms window, %d ms hop, %d channels, FFT=%d",
            window_size_ms,
            window_step_ms,
            num_channels,
            self._fft_size,
        )

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """Compute log-mel spectrogram features.

        Args:
            audio: float32 audio at 16 kHz.
                Shape (samples,) or (batch, samples).

        Returns:
            (batch, N_frames, num_channels) log-mel features.
        """
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]

        audio = audio.astype(np.float32)
        batch_size = audio.shape[0]
        results = []

        for b in range(batch_size):
            signal = audio[b]
            n_frames = compute_n_frames(
                len(signal), self.window_size_ms, self.window_step_ms, self.sample_rate
            )

            if n_frames == 0:
                results.append(
                    np.zeros((0, self.num_channels), dtype=np.float32)
                )
                continue

            # Frame the signal
            frames = np.stack(
                [
                    signal[i * self._hop_samples : i * self._hop_samples + self._window_samples]
                    for i in range(n_frames)
                ],
                axis=0,
            )  # (n_frames, window_samples)

            # Apply Hann window
            frames = frames * self._window

            # FFT → magnitude spectrum
            spectrum = np.fft.rfft(frames, n=self._fft_size)
            magnitude = np.abs(spectrum)  # (n_frames, num_fft_bins)

            # Apply mel filterbank
            mel_energies = magnitude @ self._filterbank.T  # (n_frames, num_channels)

            # Log compression
            log_mel = np.log(mel_energies + 1e-6)

            results.append(log_mel.astype(np.float32))

        return np.stack(results, axis=0)  # (batch, n_frames, num_channels)

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
