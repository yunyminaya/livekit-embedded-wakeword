# Export & Inference

The export stage converts the trained TensorFlow/Keras classifier to TFLite for deployment. The inference API provides `WakeWordModel` for stateless prediction, `StreamingWakeWordModel` for incremental processing, and `WakeWordListener` for async microphone detection.

**Source:** `src/livekit/embedded_wakeword/export/tflite.py`, `src/livekit/embedded_wakeword/inference/model.py`, `src/livekit/embedded_wakeword/inference/listener.py`
**CLI:** `livekit-wakeword export <config>`

## TFLite Export

### Classifier Export

`export_classifier()` converts the trained Keras classifier to TFLite format using `tf.lite.TFLiteConverter`.

| Property | Value |
|----------|-------|
| Input shape | `(1, 99, 40)` — spectrogram features |
| Output shape | `(1, 1)` — confidence score |
| Format | FlatBuffers (`.tflite`) |

The converter creates a concrete function with a fixed input shape for optimal TFLite compatibility.

**RepCNN reparameterization:** For `model_type: repcnn`, the export step calls `reparameterize_model()` before conversion. This fuses the multi-branch depthwise training blocks (3×3 dilated + 1×1 + identity) into single DW-Conv + PW-Conv pairs with no BatchNormalization layers, producing a compact model optimized for ESP32-S3 SIMD inference.

### INT8 Quantization

`quantize_tflite()` applies dynamic range INT8 quantization using `tf.lite.Optimize.DEFAULT`:

- Output filename: `<model_name>_int8.tflite`
- Typical size reduction: 2-4x smaller than float32

Enable via the `--quantize` flag:

```bash
livekit-wakeword export configs/hey_livekit.yaml --quantize
```

### Export Entry Point

`run_export()` loads the trained model from `output/<model_name>/<model_name>.weights.h5`, exports it to TFLite, and optionally quantizes it. Raises `FileNotFoundError` if the trained model doesn't exist.

## Inference API

**Source:** `src/livekit/embedded_wakeword/inference/model.py`, `src/livekit/embedded_wakeword/inference/listener.py`

### WakeWordModel

The `WakeWordModel` class is a stateless prediction API for wake word detection. Pass a complete audio window (~2 seconds) and receive confidence scores.

```python
from livekit.embedded_wakeword import WakeWordModel

model = WakeWordModel(models=["hey_livekit.tflite"])

# Pass ~2 seconds of 16kHz audio
scores = model.predict(audio_chunk)
# Returns: {"hey_livekit": 0.95}
```

#### Initialization

```python
WakeWordModel(
    models: list[str | Path] | None = None,  # Paths to TFLite classifiers
)
```

The `MicroFrontend` spectrogram extractor is created automatically — no external model files are needed for feature extraction.

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `predict(audio_chunk)` | `dict[str, float]` | Scores for each loaded model (0-1) |
| `load_model(path, name)` | `None` | Load additional wake word model |

#### Audio Input

- **Format:** 16kHz mono, int16 or float32
- **Chunk size:** ~2 seconds (32,000 samples) recommended — yields 99 spectrogram frames
- **Stateless:** No internal audio buffering; the caller manages the audio window

### StreamingWakeWordModel

The `StreamingWakeWordModel` class processes audio incrementally, maintaining a ring buffer of spectrogram frames.

```python
from livekit.embedded_wakeword.inference.model import StreamingWakeWordModel

model = StreamingWakeWordModel("hey_livekit.tflite")

# Process audio in small chunks (e.g., 30ms = 480 samples)
while True:
    audio_chunk = get_audio(480)  # 30ms at 16kHz
    score = model.predict_streaming(audio_chunk)
    if score is not None and score > 0.5:
        print("Wake word detected!")
```

#### How Streaming Works

1. New audio is accumulated in an internal buffer
2. When enough samples for a window (480 samples = 30ms) are available, a new spectrogram frame is computed
3. The frame is added to a ring buffer (capacity = 99 frames)
4. Once the ring buffer is full, the entire spectrogram is fed to the TFLite classifier
5. Returns a confidence score, or `None` if not enough frames yet

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `predict_streaming(audio)` | `float \| None` | Score if enough frames, else None |
| `reset()` | `None` | Clear all internal state |

### WakeWordListener

The `WakeWordListener` class provides async microphone detection with debouncing.

```python
import asyncio
from livekit.embedded_wakeword import WakeWordModel, WakeWordListener

model = WakeWordModel(models=["hey_livekit.tflite"])

async def main():
    async with WakeWordListener(model, threshold=0.5, debounce=2.0) as listener:
        while True:
            detection = await listener.wait_for_detection()
            print(f"Detected {detection.name}! ({detection.confidence:.2f})")

asyncio.run(main())
```

#### Initialization

```python
WakeWordListener(
    model: WakeWordModel,    # WakeWordModel instance with loaded classifiers
    threshold: float = 0.5,  # Detection threshold (0-1)
    debounce: float = 2.0    # Minimum seconds between detections
)
```

#### Detection Result

```python
@dataclass
class Detection:
    name: str        # Model name that triggered
    confidence: float  # Score (0-1)
    timestamp: float   # Monotonic time
```

#### Lifecycle

The listener is designed as an async context manager. On each `__aenter__`, all internal state is reset — including the audio buffer, error state, and detection queue — so the same listener instance can be safely reused across multiple `async with` blocks without stale detections carrying over.

#### Audio Capture

Uses PyAudio to capture from the default microphone:

| Parameter | Value |
|-----------|-------|
| Format | int16 (paInt16) |
| Channels | 1 (mono) |
| Sample rate | 16,000 Hz |
| Buffer size | 480 samples (30ms) |

### LiteRT Runtime

The inference code uses `ai-edge-litert` (Google's LiteRT, the successor to tflite-runtime) as the TFLite interpreter. This is a core dependency that works on all platforms.
