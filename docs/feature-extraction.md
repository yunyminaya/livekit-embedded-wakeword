# Feature Extraction Pipeline

Feature extraction converts augmented audio clips into fixed-size log-mel spectrogram arrays using a pure-numpy frontend (`MicroFrontend`), matching the micro_speech TFLite preprocessor parameters.

**Source:** `src/livekit/wakeword/data/features.py`, `src/livekit/wakeword/models/feature_extractor.py`

## Overview

```
.wav files (16kHz, ~2s)
    │
    ▼
MicroFrontend (pure numpy)
    │  30ms window, 20ms hop
    │  40-channel log-mel
    │  (batch, N_frames, 40)
    ▼
Pad/truncate to 99 frames
    │  (batch, 99, 40)
    ▼
Save as .npy
```

## MicroFrontend

**Source:** `src/livekit/wakeword/models/feature_extractor.py`

Computes log-mel spectrogram features using pure numpy operations, matching the micro_speech TFLite preprocessing model parameters used by microWakeWord.

### What is a Log-Mel Spectrogram?

A log-mel spectrogram is a visual representation of audio that mimics how humans perceive sound. The calculation proceeds in stages:

1. **Framing**: The audio is sliced into overlapping windows (30ms with 20ms hop).

2. **Windowing**: A Hann window is applied to each frame to reduce spectral leakage.

3. **FFT**: Fast Fourier Transform extracts frequency content from each windowed frame.

4. **Magnitude Spectrum**: The absolute value of the FFT output gives energy at each frequency bin.

5. **Mel Filterbank**: Triangular filters spaced on the **mel scale** are applied — a perceptual scale where equal distances sound equally spaced to humans. The mel scale: `mel = 2595 * log10(1 + f/700)`.

6. **Log Compression**: `log(mel_energies + ε)` mimics how human ears perceive loudness logarithmically.

### Parameters

| Parameter | Value |
|-----------|-------|
| Sample rate | 16,000 Hz |
| Window size | 480 samples (30ms) |
| Hop length | 320 samples (20ms) |
| FFT size | 512 (next power of 2) |
| Mel channels | 40 |
| Frequency range | 125–7,500 Hz |
| Compression | `log(x + 1e-6)` |

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

Each frame represents 20ms of audio (the hop length).

| Audio Duration | Samples | Frames | Action |
|---------------|---------|--------|--------|
| 2.0s | 32,000 | 99 | Use as-is |
| 2.5s | 40,000 | 124 | Truncate to last 99 |
| 1.5s | 24,000 | 74 | Left-pad with 25 zero frames |
| 0.5s | 8,000 | 24 | Left-pad with 75 zero frames |

### Input/Output

- **Input:** `(batch, samples)` or `(samples,)` — float32 audio at 16kHz
- **Output:** `(batch, N_frames, 40)` — log-mel features

### Mel Filterbank Construction

The filterbank is built with triangular filters spaced on the mel scale:

1. Convert lower (125 Hz) and upper (7,500 Hz) frequency limits to mel scale
2. Create `num_channels + 2` linearly spaced points on the mel scale
3. Convert back to Hz and map to FFT bin indices
4. Build triangular filters with unit-area normalization

## Frame Selection

After computing spectrograms, each clip is normalized to exactly 99 frames (for 2.0s clips):

- **99 or more frames:** Take the last 99: `spec[-99:]`
- **Fewer than 99 frames:** Left-pad with zero frames to reach 99

### Why Left-Padding?

Left-padding places zeros at the **beginning** of the sequence, so the real audio features occupy the **end** (most recent positions). This design choice has important implications:

1. **Causal alignment**: The model learns that the pattern "silence → wake word" corresponds to a detection. Real audio naturally fills from the right as new frames arrive during streaming inference.

2. **Streaming compatibility**: At inference time, you're always asking "did the wake word just happen in the most recent audio?" Having the actual audio at the end of the sequence matches this temporal expectation.

3. **Consistent anchor point**: Regardless of clip length, the wake word utterance is always anchored to the end of the feature sequence, giving the classifier a consistent reference point.

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

**Source:** `src/livekit/wakeword/data/dataset.py`

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
