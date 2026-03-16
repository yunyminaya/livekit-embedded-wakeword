"""TFLite export and INT8 quantization for wake word models."""

from __future__ import annotations

import logging
from pathlib import Path

import tensorflow as tf

from ..config import ModelType, WakeWordConfig
from ..models.pipeline import WakeWordClassifier

logger = logging.getLogger(__name__)


def export_classifier(
    config: WakeWordConfig,
    weights_path: Path,
    output_path: Path,
) -> Path:
    """Export classifier to TFLite format.

    Loads the trained Keras model weights, converts to TFLite,
    and saves as a .tflite file.

    Input shape: (1, n_frames, num_channels) — spectrogram features
    Output shape: (1, 1) — confidence score
    """
    # Build model and load weights
    model = WakeWordClassifier(config)
    n_frames = config.n_frames
    n_channels = config.augmentation.num_channels

    # Build with dummy input
    dummy = tf.zeros((1, n_frames, n_channels))
    model(dummy, training=False)
    model.load_weights(str(weights_path))

    # For RepCNN, fuse multi-branch blocks into plain conv layers
    if config.model.model_type == ModelType.repcnn:
        from ..models.classifier import reparameterize_model

        inference_model = reparameterize_model(model.classifier)
    else:
        inference_model = model

    # Create concrete function with fixed input shape
    @tf.function(input_signature=[tf.TensorSpec(shape=[1, n_frames, n_channels], dtype=tf.float32)])
    def serving_fn(x: tf.Tensor) -> tf.Tensor:
        return inference_model(x, training=False)

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [serving_fn.get_concrete_function()]
    )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_model)
    logger.info(f"Exported TFLite model to {output_path} ({len(tflite_model) / 1024:.1f} KB)")
    return output_path


def quantize_tflite(
    config: WakeWordConfig,
    weights_path: Path,
    output_path: Path | None = None,
) -> Path:
    """Export with INT8 dynamic range quantization."""
    if output_path is None:
        output_path = weights_path.with_name(
            weights_path.stem.replace(".weights", "") + "_int8.tflite"
        )

    # Build model and load weights
    model = WakeWordClassifier(config)
    n_frames = config.n_frames
    n_channels = config.augmentation.num_channels

    dummy = tf.zeros((1, n_frames, n_channels))
    model(dummy, training=False)
    model.load_weights(str(weights_path))

    # For RepCNN, fuse multi-branch blocks into plain conv layers
    if config.model.model_type == ModelType.repcnn:
        from ..models.classifier import reparameterize_model

        inference_model = reparameterize_model(model.classifier)
    else:
        inference_model = model

    @tf.function(input_signature=[tf.TensorSpec(shape=[1, n_frames, n_channels], dtype=tf.float32)])
    def serving_fn(x: tf.Tensor) -> tf.Tensor:
        return inference_model(x, training=False)

    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [serving_fn.get_concrete_function()]
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

    tflite_model = converter.convert()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_model)
    logger.info(
        f"Exported quantized TFLite model to {output_path} ({len(tflite_model) / 1024:.1f} KB)"
    )
    return output_path


def export_qat_classifier(
    config: WakeWordConfig,
    output_path: Path,
) -> Path:
    """Export a QAT-trained model to fully INT8-quantized TFLite.

    Loads the .qat.keras model, strips FakeQuant wrappers, and converts
    with INT8 input type for optimal ESP32-S3 inference.
    """
    import tensorflow_model_optimization as tfmot
    import tf_keras

    model_dir = config.model_output_dir
    qat_path = model_dir / f"{config.model_name}.qat.keras"
    if not qat_path.exists():
        raise FileNotFoundError(f"QAT model not found: {qat_path}")

    qat_model = tf_keras.models.load_model(str(qat_path))
    stripped_model = tfmot.quantization.keras.strip_quantize(qat_model)

    # Use tf_keras-native converter (from_keras_model) to avoid Keras 2/3 op mismatch
    converter = tf.lite.TFLiteConverter.from_keras_model(stripped_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.float32

    tflite_model = converter.convert()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_model)
    logger.info(
        f"Exported QAT INT8 TFLite model to {output_path} ({len(tflite_model) / 1024:.1f} KB)"
    )
    return output_path


def run_export(config: WakeWordConfig, quantize: bool = False) -> Path:
    """Export trained model to TFLite."""
    model_dir = config.model_output_dir
    weights_path = model_dir / f"{config.model_name}.weights.h5"

    if not weights_path.exists():
        raise FileNotFoundError(f"Trained model not found: {weights_path}")

    # Export classifier
    tflite_path = model_dir / f"{config.model_name}.tflite"
    export_classifier(config, weights_path, tflite_path)

    # If QAT model exists, export it as INT8
    qat_path = model_dir / f"{config.model_name}.qat.keras"
    if qat_path.exists():
        qat_tflite_path = model_dir / f"{config.model_name}_qat_int8.tflite"
        export_qat_classifier(config, qat_tflite_path)

    # Optionally quantize (dynamic range)
    if quantize:
        quantize_tflite(config, weights_path)

    return tflite_path
