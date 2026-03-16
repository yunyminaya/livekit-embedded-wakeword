"""Stateless and streaming wake word detection model using TFLite."""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path

import numpy as np

from ..models.feature_extractor import (
    SAMPLE_RATE,
    WINDOW_SIZE_MS,
    WINDOW_STEP_MS,
    MicroFrontend,
    compute_n_frames,
)

logger = logging.getLogger(__name__)

# Default frame count for 2.0s clips
DEFAULT_N_FRAMES = compute_n_frames(int(2.0 * SAMPLE_RATE))


def _load_tflite_interpreter(model_path: str | Path) -> object:
    """Load a TFLite interpreter, trying tflite-runtime first, then tensorflow."""
    model_path = str(model_path)
    try:
        from tflite_runtime.interpreter import Interpreter

        return Interpreter(model_path=model_path)
    except ImportError:
        pass
    try:
        from ai_edge_litert.interpreter import Interpreter  # type: ignore[import-untyped]

        return Interpreter(model_path=model_path)
    except ImportError:
        pass
    try:
        import tensorflow as tf

        return tf.lite.Interpreter(model_path=model_path)
    except ImportError:
        raise ImportError(
            "No TFLite runtime found. Install one of: "
            "tflite-runtime, ai-edge-litert, or tensorflow"
        )


class WakeWordModel:
    """Stateless wake word detection model using TFLite.

    The model is a pure function: pass an audio chunk (~2 seconds at 16 kHz)
    and receive confidence scores. No internal audio state is maintained.

    Example usage:
        from livekit.wakeword import WakeWordModel

        model = WakeWordModel(models=["path/to/model.tflite"])

        # Pass ~2 seconds of 16 kHz audio
        scores = model.predict(audio_chunk)
        # Returns: {"model_name": 0.95, ...}
    """

    def __init__(
        self,
        models: list[str | Path] | None = None,
    ):
        """Initialize the wake word detection model.

        Args:
            models: List of paths to wake word TFLite classifier models.
                If None, no models are loaded (call load_model() later).
        """
        self._frontend = MicroFrontend()

        # name -> (interpreter, input_index, output_index, n_frames)
        self._classifiers: dict[str, tuple[object, int, int, int]] = {}

        if models:
            for model_path in models:
                self.load_model(model_path)

    def load_model(self, model_path: str | Path, model_name: str | None = None) -> None:
        """Load a wake word classifier model.

        Args:
            model_path: Path to the TFLite wake word classifier.
            model_name: Optional name for the model. If None, derived from filename.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Wake word model not found: {model_path}")

        if model_name is None:
            model_name = model_path.stem

        interpreter = _load_tflite_interpreter(model_path)
        interpreter.allocate_tensors()  # type: ignore[union-attr]

        input_details = interpreter.get_input_details()  # type: ignore[union-attr]
        output_details = interpreter.get_output_details()  # type: ignore[union-attr]

        input_index = input_details[0]["index"]
        output_index = output_details[0]["index"]
        n_frames = input_details[0]["shape"][1]

        self._classifiers[model_name] = (interpreter, input_index, output_index, int(n_frames))
        logger.info(
            f"Loaded wake word model '{model_name}' from {model_path} "
            f"(input: 1×{n_frames}×{input_details[0]['shape'][2]})"
        )

    def predict(self, audio_chunk: np.ndarray) -> dict[str, float]:
        """Get wake word predictions for an audio chunk.

        The model is stateless — pass a complete audio window each time.
        ~2 seconds of 16 kHz audio is recommended.

        Args:
            audio_chunk: Audio samples at 16 kHz. Can be int16 or float32.

        Returns:
            Dictionary mapping model names to prediction scores (0-1).
        """
        if not self._classifiers:
            return {}

        # Convert int16 to float32 if needed
        if audio_chunk.dtype == np.int16:
            audio_chunk = audio_chunk.astype(np.float32) / 32768.0

        audio_chunk = audio_chunk.flatten()

        # Compute spectrogram: (1, actual_frames, num_channels)
        spectrogram = self._frontend(audio_chunk)
        spec = spectrogram[0]  # (actual_frames, num_channels)

        predictions = {}
        for name, (interpreter, input_idx, output_idx, n_frames) in self._classifiers.items():
            # Pad or truncate to expected n_frames
            if spec.shape[0] >= n_frames:
                input_spec = spec[-n_frames:]
            elif spec.shape[0] == 0:
                predictions[name] = 0.0
                continue
            else:
                pad = np.zeros((n_frames - spec.shape[0], spec.shape[1]), dtype=np.float32)
                input_spec = np.concatenate([pad, spec], axis=0)

            # Run TFLite inference: (1, n_frames, num_channels)
            input_data = input_spec[np.newaxis, :, :].astype(np.float32)
            interpreter.set_tensor(input_idx, input_data)  # type: ignore[union-attr]
            interpreter.invoke()  # type: ignore[union-attr]
            output = interpreter.get_tensor(output_idx)  # type: ignore[union-attr]
            score = float(output[0, 0])
            predictions[name] = score

        return predictions


class StreamingWakeWordModel:
    """Streaming wake word model that processes audio incrementally.

    Maintains a ring buffer of spectrogram frames and runs the TFLite
    classifier every time new frames are available.

    Each call to predict_streaming() processes one hop's worth of audio
    (20 ms = 320 samples by default).
    """

    def __init__(self, model_path: str | Path, model_name: str | None = None):
        model_path = Path(model_path)
        if model_name is None:
            model_name = model_path.stem
        self.model_name = model_name

        self._frontend = MicroFrontend()
        self._interpreter = _load_tflite_interpreter(model_path)
        self._interpreter.allocate_tensors()  # type: ignore[union-attr]

        input_details = self._interpreter.get_input_details()  # type: ignore[union-attr]
        output_details = self._interpreter.get_output_details()  # type: ignore[union-attr]

        self._input_index = input_details[0]["index"]
        self._output_index = output_details[0]["index"]
        self._n_frames = int(input_details[0]["shape"][1])
        self._n_features = int(input_details[0]["shape"][2])

        # Ring buffer of spectrogram frames
        self._frame_buffer: deque[np.ndarray] = deque(maxlen=self._n_frames)

        # Audio sample accumulator for partial hops
        self._audio_buffer = np.array([], dtype=np.float32)

        hop_samples = WINDOW_STEP_MS * SAMPLE_RATE // 1000
        self._hop_samples = hop_samples

    def reset(self) -> None:
        """Clear all internal state."""
        self._frame_buffer.clear()
        self._audio_buffer = np.array([], dtype=np.float32)

    def predict_streaming(self, audio: np.ndarray) -> float | None:
        """Process new audio and return a prediction if enough frames are buffered.

        Args:
            audio: New audio samples at 16 kHz (any length).

        Returns:
            Confidence score (0-1) or None if not enough frames yet.
        """
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0

        self._audio_buffer = np.concatenate([self._audio_buffer, audio.flatten()])

        # Compute new spectrogram frames from accumulated audio
        new_frames_computed = False
        window_samples = WINDOW_SIZE_MS * SAMPLE_RATE // 1000

        while len(self._audio_buffer) >= window_samples:
            window = self._audio_buffer[:window_samples]
            # Compute single frame
            spec = self._frontend(window)  # (1, 1, n_features)
            self._frame_buffer.append(spec[0, 0])
            self._audio_buffer = self._audio_buffer[self._hop_samples:]
            new_frames_computed = True

        if not new_frames_computed or len(self._frame_buffer) < self._n_frames:
            return None

        # Build spectrogram input from ring buffer
        spec_array = np.stack(list(self._frame_buffer), axis=0)  # (n_frames, n_features)
        input_data = spec_array[np.newaxis, :, :].astype(np.float32)

        self._interpreter.set_tensor(self._input_index, input_data)  # type: ignore[union-attr]
        self._interpreter.invoke()  # type: ignore[union-attr]
        output = self._interpreter.get_tensor(self._output_index)  # type: ignore[union-attr]
        return float(output[0, 0])
