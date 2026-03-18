# Architecture Overview

livekit-embedded-wakeword is a flavor of livekit-wakeword optimized for training micro wake word models for embedded devices. It uses a microWakeWord-based architecture: the TFLite Micro microfrontend (via `pymicro-features`) computes spectrogram features that feed into TensorFlow/Keras CNN classifiers (Inception, MixedNet, RepCNN, or WakeNet9), which are exported to TFLite for lightweight streaming inference on targets like ESP32-S3.

## System Architecture

**Training Pipeline:** Synthetic speech is generated via VITS TTS with SLERP speaker blending, augmented with noise/reverb, then processed through a numpy-based spectrogram frontend to produce `.npy` feature files. These features train one of four CNN classifiers (Inception, MixedNet, RepCNN, or WakeNet9), which is saved as Keras weights (`.weights.h5`).

**Inference Pipeline:** Raw 16kHz audio flows through the same spectrogram frontend (40-channel log-mel, 30ms window, 20ms hop), then through the trained classifier (exported to TFLite) to produce a detection score between 0 and 1.

## Why This Architecture?

- **Microfrontend C library** — uses the same TFLite Micro microfrontend as on-device ESPHome, via `pymicro-features`
- **TFLite inference** — lightweight runtime (~2 MB), runs on embedded devices and microcontrollers
- **Streaming support** — 30ms incremental processing via ring buffer of spectrogram frames
- **Separated dependencies** — core inference needs only numpy + tflite-runtime; training needs tensorflow; data generation needs torch (for VITS TTS only)

The CNN classifier is trained in TensorFlow/Keras for flexibility (focal loss, mixup, learning rate schedules), then exported to TFLite for deployment.

## Data Flow

### Feature Dimensions at Each Stage

| Stage | Output Shape | Description |
|-------|-------------|-------------|
| Raw audio | `(samples,)` | 16kHz float32 mono |
| MicroFrontend | `(batch, N_frames, 40)` | 40 filterbank channels, 30ms window, 20ms hop |
| Classifier | `(batch, 1)` | Sigmoid confidence score |

### Spectrogram Parameters

| Parameter | Value |
|-----------|-------|
| Sample rate | 16,000 Hz |
| Window size | 480 samples (30ms) |
| Hop length | 320 samples (20ms) |
| FFT size | 512 |
| Mel channels | 40 |
| Frequency range | 125–7,500 Hz |
| PCAN | Enabled (strength=0.95) |
| Log scale | shift=6 |
| Output scaling | uint16 (0–660) × 0.0390625 → float32 (0–26) |

### Frame Count

For a 2.0s clip (32,000 samples):

```
N_frames = floor((32000 - 480) / 320) + 1 = 99 frames
```

## Module Map

```
src/livekit/embedded_wakeword/
├── config.py                    Pydantic config models + YAML loading
├── cli.py                       Typer CLI (setup, generate, augment, train, export, eval, run)
├── models/
│   ├── feature_extractor.py     MicroFrontend (TFLite Micro microfrontend via pymicro-features)
│   ├── classifier.py            InceptionClassifier, MixedNetClassifier, RepCNNClassifier, WakeNet9Classifier
│   └── pipeline.py              WakeWordClassifier (Keras training wrapper)
├── data/
│   ├── generate.py              VITS TTS + SLERP speaker blending + adversarial negatives
│   ├── augment.py               AudioAugmentor + clip alignment (audiomentations)
│   ├── dataset.py               mmap_batch_generator + create_tf_dataset
│   └── features.py              Spectrogram extraction → .npy files
├── training/
│   ├── trainer.py               WakeWordTrainer (3-phase TF/Keras training + checkpoint averaging)
│   └── metrics.py               FPPH, recall, balanced accuracy
├── eval/
│   └── evaluate.py              DET curve, AUT, FPPH evaluation + plotting
├── export/
│   └── tflite.py                TFLite export + INT8 quantization
└── inference/
    ├── model.py                 WakeWordModel + StreamingWakeWordModel (TFLite)
    └── listener.py              WakeWordListener class (async microphone detection)
```

## Pipeline Stages

1. **[Data Generation](data-generation.md)** — Synthesize positive clips via VITS TTS with SLERP speaker blending + adversarial negatives
2. **[Augmentation](augmentation.md)** — Apply pitch shift, EQ, RIR convolution, background mixing; align clips to training windows
3. **[Feature Extraction](feature-extraction.md)** — Extract spectrogram features through TFLite Micro microfrontend → `.npy` files
4. **[Training](training.md)** — 3-phase adaptive training with focal loss, spectrogram mixup, and checkpoint averaging
5. **[Export & Inference](export-and-inference.md)** — Export classifier to TFLite and run real-time streaming detection
6. **[Evaluation](evaluation.md)** — DET curves, AUT, FPPH, and model comparison
