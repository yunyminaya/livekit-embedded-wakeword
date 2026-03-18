# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

livekit-embedded-wakeword — a flavor of livekit-wakeword optimized for training micro wake word models for embedded devices. Uses microWakeWord-based architecture: TFLite Micro microfrontend (via pymicro-features) → TensorFlow/Keras CNN classifier (Inception/MixedNet/RepCNN/WakeNet9) → TFLite export with streaming inference. VITS TTS data generation pipeline for synthetic training data. Target deployment: ESP32-S3.

## Commands

**Always use `uv` for package management. Always use `git` for version separation (branches, commits).**

```bash
# Install
uv sync --extra train --group dev    # Train + dev deps (TensorFlow)
uv sync --extra datagen --group dev  # Datagen + dev deps (torch for VITS TTS)
uv sync --group dev                  # Dev only (tests, lint)

# Test
uv run pytest tests/                 # All tests (52 tests)
uv run pytest tests/test_config.py   # Single file
uv run pytest -k "test_name"         # Single test
uv run pytest --cov=src/livekit/embedded_wakeword tests/  # With coverage

# Lint & format
uv run ruff check src/ tests/       # Lint (rules: E, F, I, UP)
uv run ruff format src/ tests/      # Auto-format
uv run mypy src/livekit/embedded_wakeword/       # Type check (strict mode)

# CLI (entry point: livekit-wakeword = livekit.embedded_wakeword.cli:app)
uv run livekit-wakeword setup            # Download VITS TTS + negative features + RIRs
uv run livekit-wakeword generate <config> # VITS TTS + SLERP speaker blending + adversarial negatives
uv run livekit-wakeword augment <config>  # Augment + extract spectrogram features → .npy
uv run livekit-wakeword train <config>    # 3-phase adaptive training (TensorFlow)
uv run livekit-wakeword export <config>   # Export classifier to TFLite
uv run livekit-wakeword eval <config>     # Evaluate model: DET curve, AUT, FPPH, recall
uv run livekit-wakeword run <config>      # Full pipeline (generate→augment→extract→train→export→eval)
```

## Architecture

### Processing Pipeline (inference)
```
Raw audio (16kHz) → MicroFrontend (pymicro-features) → Classifier (TFLite) → [0,1]
                    30ms window, 20ms hop    99×40 → 1 score
                    40-channel log-mel
```

### Processing Pipeline (training)
```
Raw audio (16kHz) → MicroFrontend (pymicro-features) → Classifier (Keras) → [0,1]
                    30ms window, 20ms hop    99×40 → 1 score
                    40-channel log-mel       Inception/MixedNet/RepCNN/WakeNet9
```

### Source Layout (`src/livekit/embedded_wakeword/`)

- **`config.py`** — Pydantic models + YAML loading (`load_config(path)`)
- **`cli.py`** — Typer CLI with all commands
- **`models/`**
  - `feature_extractor.py` — `MicroFrontend` (TFLite Micro microfrontend via pymicro-features, 30ms/20ms/40ch)
  - `classifier.py` — `InceptionClassifier` (CNN + SubSpectralNorm), `MixedNetClassifier` (MixConv + SE attention), `RepCNNClassifier` (RepVGG-style DW-sep with branch fusion), `WakeNet9Classifier` (1D CNN + LiGLU blocks), `build_classifier()` factory
  - `pipeline.py` — `WakeWordClassifier` (Keras training wrapper for classifier head)
- **`data/`**
  - `generate.py` — VITS TTS synthesis with SLERP speaker blending + adversarial negatives (requires torch)
  - `_piper_generate.py` — Vendored VITS generation from dscripka/piper-sample-generator (904-speaker SLERP)
  - `_vits_utils.py` — Vendored VITS utilities (sequence_mask, generate_path, slerp, audio_float_to_int16)
  - `augment.py` — `AudioAugmentor` (audiomentations: pitch, EQ, RIR, backgrounds); positives aligned to END of window, negatives center-padded
  - `dataset.py` — `mmap_batch_generator` (memory-mapped .npy), `create_tf_dataset()` wrapper
  - `features.py` — Extract features through MicroFrontend → .npy files
- **`training/`**
  - `trainer.py` — `WakeWordTrainer` with 3-phase training (full → refinement → fine-tuning), focal loss, adaptive negative weighting, checkpoint averaging (TF/Keras)
  - `metrics.py` — FPPH (false positives per hour), recall, balanced accuracy
- **`eval/evaluate.py`** — DET curve, AUT, FPPH evaluation + threshold optimization + plotting
- **`export/tflite.py`** — Export classifier to TFLite with optional INT8 quantization; RepCNN branch fusion via `reparameterize_model()`
- **`inference/`**
  - `model.py` — `WakeWordModel` (stateless TFLite) and `StreamingWakeWordModel` (incremental 30ms)
  - `listener.py` — `WakeWordListener` class for async microphone detection

### Key Design Decisions

- **Feature extraction uses the TFLite Micro microfrontend** C library (via `pymicro-features`), identical to the on-device ESPHome implementation. Fallback: TensorFlow `audio_microfrontend` op.
- **Spectrogram shape**: `(batch, 99, 40)` for 2.0s clips — 99 frames (30ms window, 20ms hop) × 40 filterbank channels. Features are scaled float32 (~0–26 range) via `uint16 × 0.0390625`.
- **Four classifier architectures**: Inception (multi-branch CNN + SubSpectralNorm), MixedNet (multi-kernel DW conv + SE attention), RepCNN (RepVGG-style with branch fusion for ESP32-S3), WakeNet9 (1D CNN + LiGLU, causal/streaming-friendly). Size presets (tiny→xxlarge) map to `(filters, n_blocks)`. Inception has separate presets due to ~35x higher param density.
- **Training loss**: Focal loss (gamma=2.0) with linearly increasing negative class weight + mixup + label smoothing.
- **Checkpoint averaging**: final model averages top checkpoints by 90th-pct accuracy and 10th-pct FPPH.
- **Config format**: YAML loaded via `load_config(path)`. See `configs/prod_inception.yaml` for reference. Per-architecture configs: `prod_inception.yaml`, `prod_repcnn.yaml`, `prod_mixednet.yaml`, `prod_wakenet9.yaml`.
- **Dependency separation**: Core inference needs only numpy + pymicro-features + ai-edge-litert. Training needs tensorflow. Data generation needs torch (for VITS TTS only).
- **Cloud training**: SkyPilot on Nebius (L40S GPUs). Per-architecture SkyPilot configs in `skypilot/`. Artifacts saved to `nebius://wakeword-model-bucket`.

## Documentation

For detailed documentation on each pipeline stage, see `docs/`:

- [docs/overview.md](docs/overview.md) — Architecture and data flow
- [docs/data-generation.md](docs/data-generation.md) — TTS synthesis and adversarial negatives
- [docs/augmentation.md](docs/augmentation.md) — Audio transforms and alignment
- [docs/feature-extraction.md](docs/feature-extraction.md) — Micro-speech spectrogram frontend
- [docs/training.md](docs/training.md) — 3-phase training, classifier architectures, and checkpoint averaging
- [docs/export-and-inference.md](docs/export-and-inference.md) — TFLite export (with RepCNN branch fusion) and streaming inference
- [docs/evaluation.md](docs/evaluation.md) — DET curves, AUT, FPPH, and model comparison

## Code Style

- Python 3.11+, line length 100
- Ruff for linting/formatting, mypy strict mode
- Build system: hatchling, src layout (`src/livekit/embedded_wakeword/`)
