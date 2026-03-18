# Feature Extraction Pipeline

Feature extraction converts augmented audio clips into fixed-size spectrogram arrays using the TFLite Micro microfrontend — the same C library used on-device by ESPHome's `micro_wake_word` component.

**Source:** `src/livekit/embedded_wakeword/data/features.py`, `src/livekit/embedded_wakeword/models/feature_extractor.py`

## Overview

```
.wav files (16kHz, ~2s)
    │
    ▼
MicroFrontend (pymicro-features C library)
    │  30ms window, 20ms hop
    │  40 filterbank channels
    │  PCAN + noise reduction + log scale
    │  (batch, N_frames, 40)
    ▼
Pad/truncate to 99 frames
    │  (batch, 99, 40)
    ▼
Save as .npy
```

## MicroFrontend

**Source:** `src/livekit/embedded_wakeword/models/feature_extractor.py`

Computes spectrogram features using the TFLite Micro microfrontend, matching microWakeWord's training pipeline and the on-device ESPHome implementation exactly.

### Backends

The `MicroFrontend` class supports two backends (tried in order):

| Backend | Package | Description |
|---------|---------|-------------|
| **pymicro-features** (primary) | `pymicro-features>=0.3` | CPython extension wrapping the TFLite Micro microfrontend C library |
| **TensorFlow op** (fallback) | `tensorflow>=2.15` | `audio_microfrontend` op from `tensorflow.lite.experimental.microfrontend` |

Both produce identical output. The `pymicro-features` package is recommended for inference (lightweight, no TensorFlow dependency). TensorFlow is available as a fallback during training.

### What Does the Microfrontend Compute?

The microfrontend is a lightweight signal processing pipeline (not a neural network):

1. **Windowing**: 30ms Hann-windowed frames with configurable hop (default 20ms).

2. **FFT**: Fast Fourier Transform extracts frequency content from each frame.

3. **Filterbank**: 40 triangular filters spaced on a mel-like scale from 125 Hz to 7,500 Hz.

4. **Noise Reduction**: Smoothed noise floor estimation with configurable parameters (smoothing bits=10, even=0.025, odd=0.06).

5. **PCAN Gain Control**: Per-Channel Automatic Noise-floor Cancellation normalizes each channel relative to its estimated noise floor (strength=0.95, offset=80.0).

6. **Log Scaling**: Logarithmic compression with shift=6.

7. **Output Scaling**: The C library produces uint16 values (range ~0–660). These are scaled to float32 by multiplying by `0.0390625` (= 1/25.6), giving a final range of ~0–26.

### Parameters

| Parameter | Value |
|-----------|-------|
| Sample rate | 16,000 Hz |
| Window size | 30 ms (480 samples) |
| Hop length | 20 ms (320 samples) |
| Filterbank channels | 40 |
| Frequency range | 125–7,500 Hz |
| PCAN enabled | Yes |
| PCAN strength | 0.95 |
| Noise reduction | Enabled |
| Log scale shift | 6 |
| Output type | uint16 → float32 (× 0.0390625) |
| Samples per C chunk | 160 (10ms) |

### How Frames Are Calculated

The number of output spectrogram frames depends on the audio length:

```
N_frames = floor((samples - window_size) / hop_size) + 1
         = floor((samples - 480) / 320) + 1
```

**Example**: 2 seconds of audio at 16kHz = 32,000 samples:

```
N_frames = floor((32000 - 480) / 320) + 1 = 99 frames
```

| Audio Duration | Samples | Frames | Action |
|---------------|---------|--------|--------|
| 2.0s | 32,000 | 99 | Use as-is |
| 2.5s | 40,000 | 124 | Truncate to last 99 |
| 1.5s | 24,000 | 74 | Left-pad with 25 zero frames |
| 0.5s | 8,000 | 24 | Left-pad with 75 zero frames |

### Input/Output

- **Input:** `(batch, samples)` or `(samples,)` — float32 [-1,1] or int16 audio at 16kHz
- **Output:** `(batch, N_frames, 40)` — float32 spectrogram features

Float32 input is automatically converted to int16 (matching microWakeWord convention).

### Why This Matters for Compatibility

Using the exact same microfrontend C library for both training and inference ensures:

1. **Exact match with precomputed features** — the HuggingFace `binhpham/livekit_embedded_wakeword_features` dataset was generated with this same library
2. **Exact match with on-device inference** — ESPHome's `micro_wake_word` component uses the same C code
3. **No mismatch risk** — a numpy reimplementation would produce different values due to the noise reduction, PCAN, and log scaling specifics

### Feature Scaling Convention (uint16 → float32)

The microfrontend C library internally produces **uint16** values (range ~0–660). All feature paths in this project apply the `× 0.0390625` scaling factor to convert to **float32** (range ~0–26), following the convention specified in the [precomputed features dataset](https://huggingface.co/datasets/binhpham/livekit_embedded_wakeword_features):

> *"These features are not scaled or converted to float. To do so, multiply by a factor of 0.0390625 after casting to a float."*

| Feature Source | Raw Output | Scaling Applied |
|---|---|---|
| **pymicro-features** (C extension) | float32, already scaled (~0–26) | None needed |
| **TF `audio_microfrontend`** op | uint16 (~0–660) | `× 0.0390625` after cast to float32 |
| **HuggingFace precomputed dataset** | uint16 (~0–660) in RaggedMmap | `× 0.0390625` during `_convert_ragged_to_npy` |

This ensures that all features — whether extracted live from audio or loaded from the precomputed dataset — are in the same consistent ~0–26 float32 range.

## Frame Selection

After computing spectrograms, each clip is normalized to exactly 99 frames (for 2.0s clips):

- **99 or more frames:** Take the last 99: `spec[-99:]`
- **Fewer than 99 frames:** Left-pad with zero frames to reach 99

### Why Left-Padding?

Left-padding places zeros at the **beginning** of the sequence, so the real audio features occupy the **end** (most recent positions). This design choice has important implications:

1. **Causal alignment**: The model learns that the pattern "silence → wake word" corresponds to a detection.

2. **Streaming compatibility**: At inference time, you're always asking "did the wake word just happen in the most recent audio?"

3. **Consistent anchor point**: Regardless of clip length, the wake word utterance is always anchored to the end of the feature sequence.

## Feature Extraction for Training

`run_extraction(config)` processes all four data splits:

| Clip Directory | Output File |
|---------------|-------------|
| `positive_train/` | `positive_features_train.npy` |
| `positive_test/` | `positive_features_test.npy` |
| `negative_train/` | `negative_features_train.npy` |
| `negative_test/` | `negative_features_test.npy` |

Each `.npy` file contains a float32 array of shape `(N_clips, 99, 40)`.

Audio files are read via `soundfile`, converted to float32, reduced to mono if stereo, and processed one clip at a time.

## Memory-Mapped Dataset

**Source:** `src/livekit/embedded_wakeword/data/dataset.py`

The training pipeline loads `.npy` features via memory mapping for efficient access without loading entire arrays into memory.

### mmap_batch_generator

```python
mmap_batch_generator(
    data_files: dict[str, Path],       # {"class_name": path_to_npy}
    n_per_class: dict[str, int],       # {"class_name": samples_per_batch}
    label_funcs: dict[str, Callable]   # {"class_name": label_function}
)
```

- Memory-maps all `.npy` files in read-only mode (`mmap_mode="r"`)
- Yields infinite shuffled batches with specified class composition
- Each batch is a `(features, labels)` tuple with shapes `(batch_size, 99, 40)` and `(batch_size,)`
- Wraps around each class independently using modulo arithmetic

### create_tf_dataset

Wraps `mmap_batch_generator` in a `tf.data.Dataset.from_generator()` with `.prefetch(AUTOTUNE)` for efficient TensorFlow training.

### Default Batch Composition

From `WakeWordConfig.batch_n_per_class`:

| Class | Samples per Batch |
|-------|-------------------|
| `positive` | 50 |
| `adversarial_negative` | 50 |
| `negative_speech` | 1024 |

Total batch size: 1,124 samples (50 positive + 50 adversarial + 1,024 general negative).
