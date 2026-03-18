# Evaluation

The evaluation stage runs the exported TFLite model against held-out validation data and produces a DET (Detection Error Tradeoff) curve, AUT score, and summary metrics.

**Source:** `src/livekit/embedded_wakeword/eval/evaluate.py`
**CLI:** `livekit-wakeword eval <config>` or as step 6/6 of `livekit-wakeword run`

## Overview

```
TFLite model (.tflite) + validation features (.npy)
    │
    ▼
Run inference on positive & negative samples
    │
    ▼
Compute DET curve (FPR vs FNR across thresholds)
    │
    ▼
Compute AUT (Area Under the DET curve)
    │
    ▼
Find optimal threshold (maximize recall at target FPPH)
    │
    ▼
Save DET plot (.png) + metrics (.json)
```

## CLI Usage

**Evaluate after a full pipeline run:**

```bash
uv run livekit-wakeword eval configs/hey_livekit.yaml
```

**Evaluate a specific TFLite model** (e.g., from a different training run):

```bash
uv run livekit-wakeword eval configs/hey_livekit.yaml -m /path/to/model.tflite
```

If `--model` is not specified, the default path `output/<model_name>/<model_name>.tflite` is used.

## Python API

```python
from pathlib import Path
from livekit.embedded_wakeword import load_config
from livekit.embedded_wakeword.eval.evaluate import run_eval

config = load_config("configs/hey_livekit.yaml")
model_path = Path("output/hey_livekit/hey_livekit.tflite")

results = run_eval(config, model_path)
# results = {
#     "aut": 0.0012,
#     "fpph": 0.08,
#     "recall": 0.861,
#     "accuracy": 0.93,
#     "threshold": 0.68,
#     "n_positive": 15000,
#     "n_negative": 45084,
#     "validation_hours": 25.05,
# }
```

## Metrics

### AUT — Area Under the DET Curve

The primary aggregate metric. Computed by integrating FNR as a function of FPR using the trapezoidal rule. Lower is better (0 = perfect separation).

AUT captures the full tradeoff between false positives and false negatives across all thresholds, making it useful for comparing models without committing to a specific operating point.

### DET Curve

The Detection Error Tradeoff curve plots False Positive Rate (x-axis) against False Negative Rate (y-axis) across 1001 thresholds from 0.0 to 1.0. A perfect model hugs the origin; a random classifier falls on the diagonal.

The DET curve is saved as `output/<model_name>/<model_name>_det.png` with an annotation box showing AUT, FPPH, recall, and the optimal threshold.

### FPPH — False Positives Per Hour

The number of false triggers per hour of negative audio. Computed as:

```
FPPH = count(negative_scores >= threshold) / validation_hours
```

where `validation_hours = n_negative_clips × clip_duration / 3600`.

### Recall

True positive rate at the optimal threshold:

```
Recall = mean(positive_scores >= threshold)
```

### Threshold Optimization

The evaluation uses `find_best_threshold()` to scan thresholds from 0.01 to 0.99 and select the one that **maximizes recall** while keeping **FPPH ≤ `target_fp_per_hour`** (from config). If no threshold meets the FPPH target, it falls back to maximizing balanced accuracy.

## Validation Data

### Sources

| Source | Path | Type |
|--------|------|------|
| Positive test clips | `output/<model>/positive_features_test.npy` | Wake word samples (generated + augmented) |
| Negative test clips | `output/<model>/negative_features_test.npy` | Adversarial negatives (phonetically similar) |
| General negatives | `data/features/validation_features.npy` | Precomputed spectrogram features (downloaded via `setup`) |

All feature arrays have shape `(N, 99, 40)` — N clips × 99 spectrogram frames × 40 mel channels.

### Requirements

Evaluation requires that the data generation, augmentation, and feature extraction stages have been run first (to produce the `*_features_test.npy` files). The general negative validation features are optional but recommended — without them, FPPH estimates are based only on adversarial negatives.

## Output Files

| File | Description |
|------|-------------|
| `<model_name>_det.png` | DET curve plot with metrics annotation |
| `<model_name>_eval.json` | Full metrics as JSON |

Example `_eval.json`:

```json
{
  "aut": 0.0012,
  "fpph": 0.08,
  "recall": 0.861,
  "accuracy": 0.93,
  "threshold": 0.68,
  "n_positive": 15000,
  "n_negative": 45084,
  "validation_hours": 25.05
}
```

## Comparing Models

The eval command accepts any TFLite model that takes `(1, 99, 40)` input and produces a `(1, 1)` score, making it useful for comparing models trained with different configurations:

```bash
# Evaluate an inception model
uv run livekit-wakeword eval configs/hey_livekit.yaml -m models/inception_medium.tflite

# Evaluate a mixednet model against the same validation set
uv run livekit-wakeword eval configs/hey_livekit.yaml -m models/mixednet_small.tflite
```
