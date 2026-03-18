<a href="https://livekit.io/">
  <img src="https://raw.githubusercontent.com/livekit/livekit-wakeword/main/.github/assets/livekit-mark.png" alt="LiveKit logo" width="100" height="100">
</a>

# LiveKit Embedded Wakeword

> **Note:** This is an experimental repository. APIs and architectures may change without notice.

[![CI](https://github.com/livekit/livekit-embedded-wakeword/actions/workflows/ci.yml/badge.svg)](https://github.com/livekit/livekit-embedded-wakeword/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

LiveKit Embedded Wakeword is a flavor of [livekit-wakeword](https://github.com/livekit/livekit-wakeword) optimized for training micro wake word models for embedded devices (ESP32-S3, microcontrollers). It uses a microWakeWord-based architecture — TFLite Micro microfrontend features feed directly into lightweight CNN classifiers, exported to TFLite for on-device inference.

**Features:**

- **Four classifier architectures** — Inception (multi-branch CNN), MixedNet (multi-kernel DW conv + SE attention), RepCNN (RepVGG-style with branch fusion), and WakeNet9 (1D CNN + LiGLU blocks)
- **TFLite Micro compatible** — uses the same `pymicro-features` spectrogram frontend as ESPHome's on-device implementation
- **Tiny models** — from ~3 KB (tiny) to ~256 KB (xxlarge), with INT8 quantization for further size reduction
- **Train on any platform TensorFlow supports** — local CPU/GPU or spawn [SkyPilot](https://github.com/skypilot-org/skypilot) jobs on cloud GPUs
- **End-to-end pipeline** — generate synthetic data, augment, train, export, and evaluate from a single YAML config

**Quick Links:**

- [Using Existing Models](#using-existing-models-and-library)
- [Training New Models Using The CLI](#training-new-models-using-the-cli)
- [Training New Models Using The Python API](#training-new-models-using-the-python-api)

## Quick Start

### Using Existing Models and Library

**System dependencies (for microphone listener):**

```bash
# macOS
brew install portaudio

# Ubuntu/Debian
sudo apt install portaudio19-dev
```

**Installation:**

```bash
pip install livekit-wakeword
# or
uv add livekit-wakeword
```

For microphone listening, install with the `listener` extra:

```bash
pip install livekit-wakeword[listener]
```

**Basic inference:**

```python
from livekit.embedded_wakeword import WakeWordModel

model = WakeWordModel(models=["hey_livekit.tflite"])

# Feed ~2 seconds of 16kHz audio
scores = model.predict(audio_frame)
if scores["hey_livekit"] > 0.5:
    print("Wake word detected!")
```

**Async listener with microphone:**

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

### Training New Models Using The CLI

**System dependencies:**

```bash
# macOS
brew install espeak-ng ffmpeg portaudio

# Ubuntu/Debian
sudo apt install espeak-ng libsndfile1 ffmpeg sox portaudio19-dev
```

**Installation (with pip):**

```bash
pip install livekit-wakeword[train,eval,export]
```

**Installation (with uv):**

```bash
uv tool install livekit-wakeword[train,eval,export]
```

**Installation (from source):**

```bash
# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/livekit/livekit-embedded-wakeword
cd livekit-embedded-wakeword
uv sync --all-extras
```

**Download models and data:**

```bash
livekit-wakeword setup
```

**Train a wake word:**

```bash
livekit-wakeword run configs/prod_inception.yaml
```

Or run stages individually:

```bash
livekit-wakeword generate configs/prod_inception.yaml  # TTS synthesis + adversarial negatives
livekit-wakeword augment configs/prod_inception.yaml   # Augment + extract spectrogram features
livekit-wakeword train configs/prod_inception.yaml     # 3-phase adaptive training
livekit-wakeword export configs/prod_inception.yaml    # Export to TFLite (+ INT8 quantization)
livekit-wakeword eval configs/prod_inception.yaml      # Evaluate model (DET curve, AUT, FPPH)
```

**Config:**

Per-architecture production configs are provided in `configs/`:

- `configs/prod_inception.yaml` — Inception CNN + SubSpectralNorm
- `configs/prod_repcnn.yaml` — RepCNN (recommended for ESP32-S3)
- `configs/prod_mixednet.yaml` — MixedNet + SE attention
- `configs/prod_wakenet9.yaml` — WakeNet9 (1D CNN + LiGLU)

```yaml
model_name: hey_livekit_inception
target_phrases:
  - "hey livekit"

n_samples: 25000
model:
  model_type: inception  # inception, repcnn, mixednet, or wakenet9
  model_size: xxlarge    # tiny, small, medium, large, xlarge, xxlarge
steps: 100000
max_negative_weight: 1500
target_fp_per_hour: 0.1

batch_n_per_class:
  positive: 50
  adversarial_negative: 50
  negative_speech: 128
```

**Train on cloud GPUs with SkyPilot:**

Per-architecture SkyPilot configs are in `skypilot/`:

```bash
sky launch skypilot/train_inception.yaml
sky launch skypilot/train_repcnn.yaml
sky launch skypilot/train_mixednet.yaml
sky launch skypilot/train_wakenet9.yaml
```

### Training New Models Using The Python API

The full training pipeline is available as a Python API:

```python
from livekit.embedded_wakeword import (
    WakeWordConfig,
    load_config,
    run_generate,
    run_augment,
    run_extraction,
    run_train,
    run_export,
    run_eval,
)

# Load from YAML or construct directly
config = load_config("configs/prod_inception.yaml")

# Or build a config programmatically
config = WakeWordConfig(
    model_name="hey_robot",
    target_phrases=["hey robot"],
    n_samples=5000,
    steps=30000,
)

# Run individual stages
run_generate(config)     # TTS synthesis + adversarial negatives
run_augment(config)      # Add noise, reverb, pitch shifts
run_extraction(config)   # Extract spectrogram features → .npy
run_train(config)        # 3-phase adaptive training
tflite_path = run_export(config)  # Export to TFLite

# Evaluate the exported model
results = run_eval(config, tflite_path)
print(f"AUT={results['aut']:.4f}  FPPH={results['fpph']:.2f}  Recall={results['recall']:.1%}")
```

## microWakeWord Compatibility

> **TODO:** This has not been verified yet, but exported TFLite models should be compatible with [microWakeWord](https://github.com/kahrendt/microWakeWord) and ESPHome's microWakeWord component. Both projects use the same TFLite Micro microfrontend (30ms window, 20ms hop, 40 mel channels) and the same model interface — input `(1, N_frames, 40)` spectrograms, output `(1, 1)` confidence score. Models trained here should be drop-in replacements on the inference side.

## Detailed Documentation

- [Architecture Overview](docs/overview.md) — system design and data flow
- [Data Generation](docs/data-generation.md) — TTS synthesis and adversarial negatives
- [Augmentation](docs/augmentation.md) — audio transforms and alignment
- [Feature Extraction](docs/feature-extraction.md) — micro-speech spectrogram frontend
- [Training](docs/training.md) — 3-phase training, classifier architectures, and checkpoint averaging
- [Export & Inference](docs/export-and-inference.md) — TFLite export (with RepCNN branch fusion) and streaming inference
- [Evaluation](docs/evaluation.md) — DET curves, AUT, and model comparison

## License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.
