# Training Pipeline

The training stage uses a 3-phase adaptive approach with focal loss, spectrogram mixup, AdamW optimization, and checkpoint averaging, implemented in TensorFlow/Keras.

**Source:** `src/livekit/wakeword/training/trainer.py`, `src/livekit/wakeword/training/metrics.py`
**CLI:** `livekit-wakeword train <config>`

## Overview

```
.npy features (N, 99, 40) — scaled float32, ~0–26 range
    │
    ▼
Phase 1: Full Training
    LR warmup → hold → cosine decay
    Focal loss + negative weighting + spectrogram mixup
    │
    ▼
Phase 2: Refinement
    0.1× LR, steps/10 steps
    Adaptive negative weight doubling
    │
    ▼
Phase 3: Fine-Tuning
    0.01× LR, steps/10 steps
    │
    ▼
Checkpoint Averaging
    Select top checkpoints by FPPH + recall + accuracy
    Average their weights
    │
    ▼
Threshold Optimization
    Scan 0.01–0.99 to maximize recall at target FPPH
    │
    ▼
Final model (.weights.h5)
```

## 3-Phase Training

### Phase 1 — Full Training

| Parameter | Value |
|-----------|-------|
| Steps | `config.steps` (default: 50,000) |
| Learning rate | `config.learning_rate` (default: 1e-4) |
| Optimizer | AdamW with `config.weight_decay` (default: 0.01) |
| Warmup | `steps // 5` linear warmup |
| Hold | `steps // 3` constant LR |
| Decay | Cosine decay to 0 |
| Negative weight | Linear 1.0 → `max_negative_weight` |
| Validation | Final quarter, every `steps // 20` steps |

### Phase 2 — Refinement

| Parameter | Value |
|-----------|-------|
| Steps | `config.steps // 10` |
| Learning rate | `config.learning_rate * 0.1` |
| Negative weight | `max_negative_weight` (doubled if FPPH > target) |

Between phase 1 and 2, the trainer validates and checks if FPPH exceeds `target_fp_per_hour`. If so, `max_negative_weight` is doubled to increase the penalty on false positives.

### Phase 3 — Fine-Tuning

| Parameter | Value |
|-----------|-------|
| Steps | `config.steps // 10` |
| Learning rate | `config.learning_rate * 0.01` |
| Negative weight | `max_negative_weight` (doubled again if FPPH > target) |

Between phase 2 and 3, the trainer validates again and doubles `max_negative_weight` a second time if FPPH still exceeds the target.

## Learning Rate Schedule

`_cosine_warmup_schedule(step, total_steps, warmup_steps, hold_steps, base_lr)`

```
LR
 │  ╱‾‾‾‾‾‾‾‾‾╲
 │ ╱             ╲
 │╱               ╲
 └─────────────────╲──► step
   warmup  hold     cosine decay
```

- **Warmup:** Linear from 0 to `base_lr` over `warmup_steps`
- **Hold:** Constant `base_lr` for `hold_steps`
- **Decay:** Cosine annealing from `base_lr` to 0

Phases 2 and 3 have no warmup or hold (warmup and hold steps are 0).

## Optimizer

AdamW (`tf.keras.optimizers.AdamW`) is used instead of plain Adam. AdamW decouples weight decay from the adaptive learning rate, providing proper L2 regularization. This is controlled by `config.weight_decay` (default: 0.01).

Training uses `tf.GradientTape` for explicit gradient computation and manual learning rate scheduling.

## Negative Weight Schedule

`_negative_weight_schedule(step, total_steps, max_weight)`

Linear increase from 1.0 to `max_weight` over the course of each phase:

```
weight = 1.0 + (max_weight - 1.0) * step / total_steps
```

Default `max_negative_weight` is 1500.0, meaning by the end of phase 1 the loss contribution of negative samples is weighted 1500x compared to the start.

## Loss Function

### Focal Loss

Focal loss inherently down-weights well-classified examples via a `(1 - p_t)^γ` modulating factor:

```
FL(p_t) = -(1 - p_t)^γ · log(p_t)
```

where `p_t` is the model's estimated probability for the correct class and `γ = 2.0`.

| γ value | Effect |
|---------|--------|
| 0 | Equivalent to standard BCE |
| 2 (default) | Well-classified examples (p_t > 0.9) contribute ~100x less to the loss |
| 5 | Even more aggressive down-weighting |

This eliminates the need for manual hard-example mining thresholds. Focal loss achieves the same effect smoothly and with fewer hyperparameters.

### Per-Sample Weighting

Negative samples are weighted by the current negative weight schedule value. Positive samples always have weight 1.0. The focal loss is computed per-sample, then multiplied by the per-sample weight before averaging.

## Regularization

### Label Smoothing

Training targets are softened from hard 0/1 to `ε/2` and `1 - ε/2`, where `ε = config.label_smoothing` (default: 0.05). With the default, labels become 0.025 and 0.975.

This prevents the model from producing overconfident sigmoid outputs (very close to 0.0 or 1.0), which improves score calibration and makes threshold-based detection more reliable.

### Spectrogram Mixup

During training, random pairs of samples within each batch are interpolated in spectrogram space:

```
λ ~ Beta(0.2, 0.2)
features_mixed = λ · features + (1-λ) · features[permutation]
labels_mixed = λ · labels + (1-λ) · labels[permutation]
```

The Beta(0.2, 0.2) distribution produces mixing coefficients that are usually close to 0 or 1 (light interpolation), creating virtual training examples near the original data points. This regularizes the classifier without requiring changes to the audio augmentation pipeline.

## Validation

### Data Sources

- **Positive:** `positive_features_test.npy`
- **Negative:** `negative_features_test.npy` + optional `validation_features.npy` (from microWakeWord precomputed features)

### Metrics

**Source:** `src/livekit/wakeword/training/metrics.py`

| Metric | Function | Description |
|--------|----------|-------------|
| FPPH | `false_positives_per_hour()` | Count of predictions >= threshold on negatives, divided by total hours |
| Recall | `recall_at_threshold()` | True positive rate: `mean(positive_preds >= threshold)` |
| Balanced Accuracy | `accuracy()` | `(TPR + TNR) / 2` at the given threshold |

The `evaluate_model()` function computes all metrics at once. Validation hours are computed from the actual negative clip count × clip duration (default 2.0s), not a hardcoded value.

### Validation Schedule

Validation runs during the final quarter of each phase, at intervals of `steps // 20`. Each validation produces a checkpoint.

## Checkpoint Averaging

After all three phases, the best checkpoints are averaged to produce the final model.

### Selection Criteria

A checkpoint qualifies if **all three** conditions are met:

| Metric | Threshold |
|--------|-----------|
| FPPH | <= 10th percentile of all checkpoints |
| Recall | >= 90th percentile of all checkpoints |
| Balanced Accuracy | >= 90th percentile of all checkpoints |

If no checkpoints meet all three criteria, the checkpoint with the highest recall is used as a fallback.

### Weight Averaging

For qualifying checkpoints, each weight array is stacked and averaged:

```python
averaged[i] = mean(stack([ckpt.weights[i] for ckpt in selected]))
```

This produces a smoother model that generalizes better than any single checkpoint.

## Threshold Optimization

After checkpoint averaging, the trainer searches for the optimal detection threshold on the validation set using `find_best_threshold()`:

1. Scan thresholds from 0.01 to 0.99 in steps of 0.01
2. For each threshold, compute FPPH and recall
3. Select the threshold that **maximizes recall** while keeping **FPPH ≤ `target_fp_per_hour`**
4. If no threshold meets the FPPH target, fall back to the one with the highest balanced accuracy

The optimal threshold and its metrics are logged to `metrics.json` with `"note": "optimal_threshold"`. This threshold can be used at inference time instead of the default 0.5.

## Training Data Sources

The dataset loads features from:

| Class | Source File | Label |
|-------|-----------|-------|
| `positive` | `positive_features_train.npy` | 1 |
| `adversarial_negative` | `negative_features_train.npy` | 0 |
| `negative_speech` | `data/features/speech_features.npy` | 0 |

The negative speech features (if available) provide general audio spectrograms as additional negative examples, downloaded from the `binhpham/livekit_embedded_wakeword_features` HuggingFace dataset via `livekit-wakeword setup`. Raw uint16 values from the dataset are scaled by `× 0.0390625` to float32 during conversion (see [Feature Extraction — Scaling Convention](feature-extraction.md#feature-scaling-convention-uint16--float32)).

## Default Training Configuration

| Field | Default |
|-------|---------|
| `steps` | 50,000 |
| `learning_rate` | 1e-4 |
| `weight_decay` | 0.01 |
| `label_smoothing` | 0.05 |
| `max_negative_weight` | 1500.0 |
| `target_fp_per_hour` | 0.2 |
| `batch_n_per_class.positive` | 50 |
| `batch_n_per_class.adversarial_negative` | 50 |
| `batch_n_per_class.negative_speech` | 1024 |

## Classifier Architectures

Two classifier types are available, both taking input `(batch, 99, 40)` spectrogram features and producing `(batch, 1)` scores.

### Inception (`model_type: inception`)

CNN architecture with Sub-Spectral Normalization and Inception-style parallel branches, inspired by microWakeWord.

```
Input (99, 40) → expand to (99, 40, 1)
→ CNN1 blocks: Conv2D(5×5) + SubSpectralNorm + ReLU
→ Inception blocks: 3 branches (1×1, dilated K×F, double K×F) → concat → 1×1 project
→ GlobalAveragePool → Dropout → Dense(1, sigmoid)
```

Sub-Spectral Normalization splits the frequency axis into 5 sub-bands and applies BatchNorm independently to each, helping the model learn frequency-band-specific normalization.

### MixedNet (`model_type: mixednet`)

MixConv architecture with multi-kernel depthwise convolutions and optional squeeze-and-excitation attention.

```
Input (99, 40) → expand to (99, 40, 1)
→ Conv2D(3×3) + BatchNorm + ReLU
→ MixConv blocks: split channels → different-kernel DW conv (3,5,7,9) → concat → 1×1 project
→ Optional SE attention on later blocks
→ GlobalAveragePool → Dropout → Dense(1, sigmoid)
```

### Size Presets

| Size | filters | n_blocks |
|------|---------|----------|
| tiny | 8 | 2 |
| small | 16 | 3 |
| medium | 24 | 5 |
| large | 32 | 7 |

## Output

The trained model weights are saved to `output/<model_name>/<model_name>.weights.h5`.
