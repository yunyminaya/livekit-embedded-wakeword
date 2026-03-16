"""3-phase adaptive training loop for wake word classifiers (TensorFlow/Keras)."""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import TypedDict

import numpy as np
import tensorflow as tf

from ..config import WakeWordConfig
from ..data.dataset import create_tf_dataset
from ..models.pipeline import WakeWordClassifier
from ..utils import configure_gpu
from .metrics import evaluate_model, find_best_threshold

logger = logging.getLogger(__name__)


class _Checkpoint(TypedDict):
    step: int
    phase: int
    metrics: dict[str, float]
    weights: list[np.ndarray]


def _cosine_warmup_schedule(
    step: int,
    total_steps: int,
    warmup_steps: int,
    hold_steps: int,
    base_lr: float,
) -> float:
    """Learning rate with warmup → hold → cosine decay."""
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    elif step < warmup_steps + hold_steps:
        return base_lr
    else:
        decay_steps = total_steps - warmup_steps - hold_steps
        progress = (step - warmup_steps - hold_steps) / max(1, decay_steps)
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def _negative_weight_schedule(step: int, total_steps: int, max_weight: float) -> float:
    """Linear schedule for negative class weight: 1 → max_weight."""
    return 1.0 + (max_weight - 1.0) * step / max(1, total_steps)


def focal_loss(
    predictions: tf.Tensor,
    targets: tf.Tensor,
    gamma: float = 2.0,
) -> tf.Tensor:
    """Focal loss for binary classification (per-sample, unreduced).

    Down-weights well-classified examples by (1 - p_t)^gamma, automatically
    focusing training on hard examples near the decision boundary.

    Args:
        predictions: Model output probabilities (after sigmoid), shape (N,).
        targets: Ground truth labels 0 or 1, shape (N,).
        gamma: Focusing parameter. 0 = standard BCE, 2 = typical focal loss.

    Returns:
        Per-sample focal loss, shape (N,).
    """
    # Cast to float32 for numerical stability (mixed precision outputs float16)
    predictions = tf.cast(predictions, tf.float32)
    targets = tf.cast(targets, tf.float32)
    # Clip to avoid log(0)
    eps = 1e-7
    predictions = tf.clip_by_value(predictions, eps, 1.0 - eps)
    bce = -targets * tf.math.log(predictions) - (1 - targets) * tf.math.log(1 - predictions)
    p_t = predictions * targets + (1 - predictions) * (1 - targets)
    focal_weight = tf.pow(1 - p_t, gamma)
    return focal_weight * bce


class WakeWordTrainer:
    """Three-phase adaptive trainer for wake word classifiers.

    Phase 1: Full training with warmup + cosine decay
    Phase 2: Refinement at lower LR, adaptive negative weight
    Phase 3: Fine-tuning at lowest LR
    """

    def __init__(self, config: WakeWordConfig):
        configure_gpu()
        self.config = config
        self.model = WakeWordClassifier(config)
        # Build model with dummy input to initialize weights
        n_frames = config.n_frames
        n_channels = config.augmentation.num_channels
        dummy = tf.zeros((1, n_frames, n_channels))
        self.model(dummy, training=False)
        self.checkpoints: list[_Checkpoint] = []
        self._metrics_log: list[dict[str, object]] = []
        self._metrics_path = config.model_output_dir / f"{config.model_name}_metrics.json"
        self._train_start: float = 0.0

        # Compile the training step for GPU acceleration (XLA for kernel fusion)
        self._compiled_train_step = tf.function(self._train_step, jit_compile=True)

    def _build_dataset(self) -> tf.data.Dataset:
        model_dir = self.config.model_output_dir
        data_files: dict[str, str | Path] = {
            "positive": model_dir / "positive_features_train.npy",
            "adversarial_negative": model_dir / "negative_features_train.npy",
        }
        # Add negative speech features if available
        neg_speech_path = self.config.data_path / "features" / "speech_features.npy"
        if neg_speech_path.exists():
            data_files["negative_speech"] = neg_speech_path
        else:
            logger.warning(
                "Negative speech features not found at %s. Training without general "
                "negative speech data — the model may have a high false positive rate. "
                "Run setup without --skip-negatives to download.",
                neg_speech_path,
            )

        n_frames = self.config.n_frames
        n_channels = self.config.augmentation.num_channels

        return create_tf_dataset(
            data_files=data_files,
            n_per_class=self.config.batch_n_per_class,
            label_funcs={
                "positive": lambda _: 1,
                "adversarial_negative": lambda _: 0,
                "negative_speech": lambda _: 0,
            },
            n_frames=n_frames,
            num_channels=n_channels,
        )

    def _load_validation_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Load test features for validation."""
        model_dir = self.config.model_output_dir
        n_frames = self.config.n_frames
        n_channels = self.config.augmentation.num_channels
        empty = np.zeros((0, n_frames, n_channels))

        pos_path = model_dir / "positive_features_test.npy"
        neg_path = model_dir / "negative_features_test.npy"

        pos = np.load(str(pos_path)) if pos_path.exists() else empty
        neg = np.load(str(neg_path)) if neg_path.exists() else empty

        # Also load validation features if available
        val_path = self.config.data_path / "features" / "validation_features.npy"
        if val_path.exists():
            val_neg = np.load(str(val_path))
            if val_neg.ndim == 3 and val_neg.shape[1:] == (n_frames, n_channels):
                neg = np.concatenate([neg, val_neg], axis=0) if neg.shape[0] > 0 else val_neg
            else:
                logger.warning(
                    "Validation features shape %s doesn't match expected (%d, %d), skipping",
                    val_neg.shape,
                    n_frames,
                    n_channels,
                )

        return pos, neg

    def _predict(self, features: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """Run model prediction on numpy features."""
        all_preds: list[np.ndarray] = []
        for i in range(0, len(features), batch_size):
            batch = features[i : i + batch_size]
            preds = self.model(batch, training=False).numpy()
            all_preds.append(preds)
        return np.concatenate(all_preds, axis=0).squeeze(-1)

    def _validate(self) -> dict[str, float]:
        """Run validation and return metrics."""
        pos_features, neg_features = self._load_validation_data()
        if pos_features.shape[0] == 0:
            return {"fpph": 0.0, "recall": 0.0, "accuracy": 0.0, "threshold": 0.5}
        pos_preds = self._predict(pos_features)
        neg_preds = self._predict(neg_features) if neg_features.shape[0] > 0 else np.array([])

        # Compute actual validation hours from clip count × clip duration
        clip_duration = self.config.augmentation.clip_duration
        validation_hours = neg_features.shape[0] * clip_duration / 3600.0
        return evaluate_model(
            pos_preds,
            neg_preds,
            threshold=0.5,
            validation_hours=validation_hours,
        )

    def _log_metrics(self, step: int, phase: int, metrics: dict[str, float]) -> None:
        """Append a metrics entry and flush to disk."""
        entry: dict[str, object] = {
            "step": step,
            "phase": phase,
            "elapsed_s": round(time.time() - self._train_start, 1),
            **metrics,
        }
        self._metrics_log.append(entry)
        self._metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self._metrics_path.write_text(json.dumps(self._metrics_log, indent=2) + "\n")

    def _save_checkpoint(self, step: int, phase: int, metrics: dict[str, float]) -> None:
        """Save checkpoint if it meets quality criteria."""
        self.checkpoints.append(
            {
                "step": step,
                "phase": phase,
                "metrics": metrics,
                "weights": [w.numpy().copy() for w in self.model.weights],
            }
        )

    def _train_step(
        self,
        features: tf.Tensor,
        labels: tf.Tensor,
        optimizer: tf.keras.optimizers.Optimizer,
        neg_weight: tf.Tensor,
        lam: tf.Tensor,
        label_smoothing: tf.Tensor,
    ) -> tf.Tensor:
        """Single compiled training step (runs on GPU via tf.function)."""
        # Mixup
        perm = tf.random.shuffle(tf.range(tf.shape(features)[0]))
        features = lam * features + (1 - lam) * tf.gather(features, perm)
        labels = lam * labels + (1 - lam) * tf.gather(labels, perm)

        # Label smoothing
        labels = labels * (1 - label_smoothing) + 0.5 * label_smoothing

        with tf.GradientTape() as tape:
            predictions = tf.squeeze(self.model(features, training=True), axis=-1)
            loss_per_sample = focal_loss(predictions, labels, gamma=2.0)
            weights = tf.where(labels < 0.5, neg_weight, 1.0)
            loss = tf.reduce_mean(loss_per_sample * weights)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def _train_phase(
        self,
        phase: int,
        steps: int,
        base_lr: float,
        max_negative_weight: float,
        dataset: tf.data.Dataset,
    ) -> None:
        """Run a single training phase."""
        logger.info(
            f"=== Phase {phase}: {steps} steps, LR={base_lr}, max_neg_w={max_negative_weight} ==="
        )

        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=base_lr,
            weight_decay=self.config.weight_decay,
        )
        # Pre-build optimizer variables so tf.function doesn't hit
        # "singleton tf.Variables created on the first call" error
        optimizer.build(self.model.trainable_variables)
        label_smoothing_val = self.config.label_smoothing
        label_smoothing_t = tf.constant(label_smoothing_val, dtype=tf.float32)

        warmup_steps = steps // 5 if phase == 1 else 0
        hold_steps = steps // 3 if phase == 1 else 0

        from tqdm import tqdm

        data_iter = iter(dataset)
        validation_interval = max(1, steps // 20)
        log_interval = max(1, steps // 100)

        pbar = tqdm(range(steps), desc=f"Phase {phase}", unit="step")
        last_loss = 0.0
        for step in pbar:
            # Get batch
            try:
                features, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(dataset)
                features, labels = next(data_iter)

            # Schedule values
            neg_weight = _negative_weight_schedule(step, steps, max_negative_weight)
            lr = _cosine_warmup_schedule(step, steps, warmup_steps, hold_steps, base_lr)
            optimizer.learning_rate.assign(lr)

            lam = tf.constant(np.random.beta(0.2, 0.2), dtype=tf.float32)
            neg_weight_t = tf.constant(neg_weight, dtype=tf.float32)

            # Compiled training step (runs on GPU)
            loss = self._compiled_train_step(
                features, labels, optimizer, neg_weight_t, lam, label_smoothing_t
            )

            # Only sync loss to CPU periodically (avoids GPU stall every step)
            if step % log_interval == 0:
                last_loss = float(loss.numpy())
            pbar.set_postfix(loss=f"{last_loss:.4f}", lr=f"{lr:.1e}", neg_w=f"{neg_weight:.0f}")

            # Validation in final quarter
            if step >= steps * 3 // 4 and step % validation_interval == 0:
                metrics = self._validate()
                pbar.write(
                    f"  Validation @ step {step}: "
                    f"FPPH={metrics['fpph']:.2f}, "
                    f"Recall={metrics['recall']:.3f}, Acc={metrics['accuracy']:.3f}"
                )
                self._log_metrics(step, phase, metrics)
                self._save_checkpoint(step, phase, metrics)

    def train(self) -> tf.keras.Model:
        """Run full 3-phase training. Returns the final averaged model."""
        self._train_start = time.time()
        dataset = self._build_dataset()
        steps = self.config.steps
        max_neg_w = self.config.max_negative_weight

        # Phase 1: Full training
        self._train_phase(
            phase=1,
            steps=steps,
            base_lr=self.config.learning_rate,
            max_negative_weight=max_neg_w,
            dataset=dataset,
        )

        # Phase 2: Refinement
        metrics = self._validate()
        self._log_metrics(steps, 1, metrics)
        if metrics["fpph"] > self.config.target_fp_per_hour:
            max_neg_w *= 2
            logger.info(f"FP rate too high, doubling max_negative_weight to {max_neg_w}")

        self._train_phase(
            phase=2,
            steps=steps // 10,
            base_lr=self.config.learning_rate * 0.1,
            max_negative_weight=max_neg_w,
            dataset=dataset,
        )

        # Phase 3: Fine-tuning
        metrics = self._validate()
        self._log_metrics(steps + steps // 10, 2, metrics)
        if metrics["fpph"] > self.config.target_fp_per_hour:
            max_neg_w *= 2
            logger.info(f"FP rate still high, doubling max_negative_weight to {max_neg_w}")

        self._train_phase(
            phase=3,
            steps=steps // 10,
            base_lr=self.config.learning_rate * 0.01,
            max_negative_weight=max_neg_w,
            dataset=dataset,
        )

        # Final validation after phase 3
        metrics = self._validate()
        total_steps = steps + steps // 10 + steps // 10
        self._log_metrics(total_steps, 3, metrics)

        # Select and average best checkpoints
        self._average_best_checkpoints()

        # Log final averaged model metrics
        final_metrics = self._validate()
        self._log_metrics(total_steps, 0, {**final_metrics, "note": "final_averaged"})  # type: ignore[arg-type]

        # Optimize detection threshold on validation set
        optimal = self._find_optimal_threshold()
        self._log_metrics(total_steps, 0, {**optimal, "note": "optimal_threshold"})  # type: ignore[arg-type]
        logger.info(
            f"Optimal threshold: {optimal['threshold']:.2f} "
            f"(FPPH={optimal['fpph']:.2f}, Recall={optimal['recall']:.3f})"
        )
        logger.info(f"Metrics saved to {self._metrics_path}")

        return self.model

    def _find_optimal_threshold(self) -> dict[str, float]:
        """Find best detection threshold on validation data."""
        pos_features, neg_features = self._load_validation_data()
        if pos_features.shape[0] == 0:
            return {"fpph": 0.0, "recall": 0.0, "accuracy": 0.0, "threshold": 0.5}
        pos_preds = self._predict(pos_features)
        neg_preds = self._predict(neg_features) if neg_features.shape[0] > 0 else np.array([])

        clip_duration = self.config.augmentation.clip_duration
        validation_hours = neg_features.shape[0] * clip_duration / 3600.0
        return find_best_threshold(
            pos_preds,
            neg_preds,
            validation_hours=validation_hours,
            target_fpph=self.config.target_fp_per_hour,
        )

    def _average_best_checkpoints(self) -> None:
        """Average weights of top checkpoints.

        Select models in 90th percentile accuracy/recall and 10th percentile FP rate.
        """
        if not self.checkpoints:
            logger.warning("No checkpoints saved, returning current model")
            return

        # Extract metrics
        fpph_values = [c["metrics"]["fpph"] for c in self.checkpoints]
        recall_values = [c["metrics"]["recall"] for c in self.checkpoints]
        acc_values = [c["metrics"]["accuracy"] for c in self.checkpoints]

        # Filter: low FP (10th percentile) + high recall/accuracy (90th percentile)
        fp_threshold = np.percentile(fpph_values, 10) if len(fpph_values) > 1 else float("inf")
        recall_threshold = np.percentile(recall_values, 90) if len(recall_values) > 1 else 0.0
        acc_threshold = np.percentile(acc_values, 90) if len(acc_values) > 1 else 0.0

        selected = [
            c
            for c in self.checkpoints
            if c["metrics"]["fpph"] <= fp_threshold
            and c["metrics"]["recall"] >= recall_threshold
            and c["metrics"]["accuracy"] >= acc_threshold
        ]

        if not selected:
            # Fallback: use checkpoint with best recall
            selected = [max(self.checkpoints, key=lambda c: c["metrics"]["recall"])]

        logger.info(f"Averaging {len(selected)} checkpoints")

        # Average weights
        avg_weights = []
        for i in range(len(selected[0]["weights"])):
            stacked = np.stack([c["weights"][i] for c in selected], axis=0)
            avg_weights.append(stacked.mean(axis=0))

        self.model.set_weights(avg_weights)

    def save(self, path: Path) -> None:
        """Save trained model weights."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(str(path))
        logger.info(f"Saved model weights to {path}")


def run_train(config: WakeWordConfig) -> Path:
    """Run training and return path to saved model."""
    trainer = WakeWordTrainer(config)
    trainer.train()

    model_path = config.model_output_dir / f"{config.model_name}.weights.h5"
    trainer.save(model_path)
    return model_path
