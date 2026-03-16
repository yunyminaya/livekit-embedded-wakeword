"""TensorFlow/Keras classification heads for wake word detection.

Implements Inception and MixedNet CNN architectures inspired by microWakeWord.
Input: (batch, N_frames, N_features) log-mel spectrogram features.
Output: (batch, 1) confidence score in [0, 1].
"""

from __future__ import annotations

import tensorflow as tf

from ..config import MODEL_SIZE_PRESETS, ModelSize, ModelType


class SubSpectralNormalization(tf.keras.layers.Layer):  # type: ignore[misc]
    """Sub-Spectral Normalization: split frequency axis into sub-bands and BatchNorm each.

    This helps the model learn frequency-band-specific normalization,
    improving robustness to spectral variations.
    """

    def __init__(self, num_sub_bands: int = 5, **kwargs: object):
        super().__init__(**kwargs)
        self.num_sub_bands = num_sub_bands
        self._batch_norms: list[tf.keras.layers.BatchNormalization] = []

    def build(self, input_shape: tf.TensorShape) -> None:
        self._batch_norms = [
            tf.keras.layers.BatchNormalization(name=f"ssn_bn_{i}")
            for i in range(self.num_sub_bands)
        ]
        super().build(input_shape)

    def call(
        self, x: tf.Tensor, training: bool | None = None
    ) -> tf.Tensor:
        # x: (batch, time, freq, channels)
        sub_bands = tf.split(x, self.num_sub_bands, axis=2)
        normalized = [
            bn(sb, training=training)
            for bn, sb in zip(self._batch_norms, sub_bands)
        ]
        return tf.concat(normalized, axis=2)

    def get_config(self) -> dict[str, object]:
        config = super().get_config()
        config["num_sub_bands"] = self.num_sub_bands
        return config


class InceptionBlock(tf.keras.layers.Layer):  # type: ignore[misc]
    """Inception-style block with 3 parallel branches.

    Branch 1: 1x1 convolution (pointwise)
    Branch 2: KxF convolution with optional dilation
    Branch 3: Double KxF convolution (deeper receptive field)

    All branches are concatenated then projected back via 1x1 conv.
    """

    def __init__(
        self,
        filters: int,
        kernel_time: int = 5,
        kernel_freq: int = 3,
        dilation: int = 1,
        **kwargs: object,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_time = kernel_time
        self.kernel_freq = kernel_freq
        self.dilation = dilation

    def build(self, input_shape: tf.TensorShape) -> None:
        f = self.filters
        kt, kf = self.kernel_time, self.kernel_freq

        # Branch 1: pointwise
        self.branch1_conv = tf.keras.layers.Conv2D(f, (1, 1), padding="same")

        # Branch 2: dilated KxF conv
        self.branch2_conv = tf.keras.layers.Conv2D(
            f, (kt, kf), padding="same", dilation_rate=(self.dilation, 1)
        )

        # Branch 3: double KxF conv
        self.branch3_conv1 = tf.keras.layers.Conv2D(f, (kt, kf), padding="same")
        self.branch3_conv2 = tf.keras.layers.Conv2D(f, (kt, kf), padding="same")

        # Merge
        self.pointwise = tf.keras.layers.Conv2D(f, (1, 1), padding="same")
        self.ssn = SubSpectralNormalization(num_sub_bands=5)
        self.relu = tf.keras.layers.ReLU()

        # Residual projection if needed
        in_channels = input_shape[-1]
        self._needs_projection = in_channels != f
        if self._needs_projection:
            self.residual_proj = tf.keras.layers.Conv2D(f, (1, 1), padding="same")

        super().build(input_shape)

    def call(
        self, x: tf.Tensor, training: bool | None = None
    ) -> tf.Tensor:
        b1 = self.branch1_conv(x)
        b2 = self.branch2_conv(x)
        b3 = self.branch3_conv2(self.branch3_conv1(x))

        concat = tf.concat([b1, b2, b3], axis=-1)
        out = self.pointwise(concat)
        out = self.ssn(out, training=training)
        out = self.relu(out)

        # Residual connection
        residual = self.residual_proj(x) if self._needs_projection else x
        return out + residual

    def get_config(self) -> dict[str, object]:
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_time": self.kernel_time,
            "kernel_freq": self.kernel_freq,
            "dilation": self.dilation,
        })
        return config


class InceptionClassifier(tf.keras.Model):  # type: ignore[misc]
    """Inception-based wake word classifier (inspired by microWakeWord).

    Architecture:
        Input (N_frames, N_features) → reshape to (N_frames, N_features, 1)
        → CNN1 blocks (Conv2D + SubSpectralNorm + ReLU)
        → Inception blocks (3 parallel branches)
        → GlobalAveragePooling2D → Dropout → Dense(1, sigmoid)
    """

    def __init__(
        self,
        n_frames: int = 99,
        n_features: int = 40,
        filters: int = 16,
        n_cnn1_blocks: int = 2,
        n_inception_blocks: int = 3,
        dropout: float = 0.1,
        **kwargs: object,
    ):
        super().__init__(**kwargs)
        self.n_frames = n_frames
        self.n_features = n_features
        self._filters = filters

        # CNN1 blocks: initial feature extraction
        self.cnn1_blocks: list[tuple[
            tf.keras.layers.Conv2D,
            SubSpectralNormalization,
            tf.keras.layers.ReLU,
        ]] = []
        for i in range(n_cnn1_blocks):
            conv = tf.keras.layers.Conv2D(
                filters, (5, 5), padding="same", name=f"cnn1_conv_{i}"
            )
            ssn = SubSpectralNormalization(num_sub_bands=5, name=f"cnn1_ssn_{i}")
            relu = tf.keras.layers.ReLU(name=f"cnn1_relu_{i}")
            self.cnn1_blocks.append((conv, ssn, relu))

        # Inception blocks
        self.inception_blocks: list[InceptionBlock] = []
        for i in range(n_inception_blocks):
            dilation = 1 + (i % 3)  # cycle through 1, 2, 3
            block = InceptionBlock(
                filters=filters, dilation=dilation, name=f"inception_{i}"
            )
            self.inception_blocks.append(block)

        # Classification head
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.dense_out = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(
        self, x: tf.Tensor, training: bool | None = None
    ) -> tf.Tensor:
        """Forward pass.

        Args:
            x: (batch, N_frames, N_features) spectrogram features

        Returns:
            (batch, 1) confidence score [0, 1]
        """
        # Reshape to (batch, time, freq, 1) for Conv2D
        x = tf.expand_dims(x, axis=-1)

        # CNN1 blocks
        for conv, ssn, relu in self.cnn1_blocks:
            x = relu(ssn(conv(x), training=training))

        # Inception blocks
        for block in self.inception_blocks:
            x = block(x, training=training)

        # Classification head
        x = self.global_pool(x)
        x = self.dropout(x, training=training)
        return self.dense_out(x)

    def get_config(self) -> dict[str, object]:
        config = super().get_config()
        config.update({
            "n_frames": self.n_frames,
            "n_features": self.n_features,
            "filters": self._filters,
        })
        return config


class MixConvBlock(tf.keras.layers.Layer):  # type: ignore[misc]
    """Mixed Depthwise Convolution block.

    Splits input channels into groups, applies different-sized
    depthwise convolutions per group, concatenates, then applies
    pointwise 1x1 convolution.
    """

    def __init__(
        self,
        filters: int,
        num_groups: int = 4,
        use_attention: bool = False,
        **kwargs: object,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.num_groups = num_groups
        self.use_attention = use_attention

    def build(self, input_shape: tf.TensorShape) -> None:
        in_channels = int(input_shape[-1])
        self._group_size = in_channels // self.num_groups
        actual_groups = min(self.num_groups, in_channels)

        # Different kernel sizes for each group
        kernel_sizes = [3, 5, 7, 9][:actual_groups]
        self.dw_convs = [
            tf.keras.layers.DepthwiseConv2D(
                (k, 1), padding="same", depth_multiplier=1, name=f"dw_conv_{k}"
            )
            for k in kernel_sizes
        ]

        self.pointwise = tf.keras.layers.Conv2D(self.filters, (1, 1), padding="same")
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

        self._needs_projection = in_channels != self.filters
        if self._needs_projection:
            self.residual_proj = tf.keras.layers.Conv2D(
                self.filters, (1, 1), padding="same"
            )

        # Optional channel attention (CBAM-style squeeze-and-excitation)
        if self.use_attention:
            self.se_pool = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)
            self.se_fc1 = tf.keras.layers.Dense(
                max(1, self.filters // 4), activation="relu"
            )
            self.se_fc2 = tf.keras.layers.Dense(self.filters, activation="sigmoid")

        super().build(input_shape)

    def call(
        self, x: tf.Tensor, training: bool | None = None
    ) -> tf.Tensor:
        groups = tf.split(x, len(self.dw_convs), axis=-1)
        mixed = [conv(g) for conv, g in zip(self.dw_convs, groups)]
        out = tf.concat(mixed, axis=-1)
        out = self.pointwise(out)
        out = self.bn(out, training=training)
        out = self.relu(out)

        # Squeeze-and-excitation attention
        if self.use_attention:
            se = self.se_pool(out)
            se = self.se_fc1(se)
            se = self.se_fc2(se)
            out = out * se

        # Residual
        residual = self.residual_proj(x) if self._needs_projection else x
        return out + residual

    def get_config(self) -> dict[str, object]:
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "num_groups": self.num_groups,
            "use_attention": self.use_attention,
        })
        return config


class MixedNetClassifier(tf.keras.Model):  # type: ignore[misc]
    """MixedNet-based wake word classifier (inspired by microWakeWord).

    Architecture:
        Input (N_frames, N_features) → reshape to (N_frames, N_features, 1)
        → Initial Conv2D
        → MixConv blocks with optional attention
        → GlobalAveragePooling2D → Dropout → Dense(1, sigmoid)
    """

    def __init__(
        self,
        n_frames: int = 99,
        n_features: int = 40,
        filters: int = 32,
        n_blocks: int = 3,
        num_groups: int = 4,
        use_attention: bool = True,
        dropout: float = 0.1,
        **kwargs: object,
    ):
        super().__init__(**kwargs)
        self.n_frames = n_frames
        self.n_features = n_features
        self._filters = filters

        # Initial conv to expand channels
        self.init_conv = tf.keras.layers.Conv2D(
            filters, (3, 3), padding="same", name="init_conv"
        )
        self.init_bn = tf.keras.layers.BatchNormalization(name="init_bn")
        self.init_relu = tf.keras.layers.ReLU(name="init_relu")

        # MixConv blocks
        self.mix_blocks: list[MixConvBlock] = []
        for i in range(n_blocks):
            # Enable attention on later blocks
            use_attn = use_attention and (i >= n_blocks // 2)
            actual_groups = min(num_groups, filters)
            block = MixConvBlock(
                filters=filters,
                num_groups=actual_groups,
                use_attention=use_attn,
                name=f"mixconv_{i}",
            )
            self.mix_blocks.append(block)

        # Classification head
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.dense_out = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(
        self, x: tf.Tensor, training: bool | None = None
    ) -> tf.Tensor:
        """Forward pass.

        Args:
            x: (batch, N_frames, N_features) spectrogram features

        Returns:
            (batch, 1) confidence score [0, 1]
        """
        # Reshape to (batch, time, freq, 1) for Conv2D
        x = tf.expand_dims(x, axis=-1)

        # Initial convolution
        x = self.init_relu(self.init_bn(self.init_conv(x), training=training))

        # MixConv blocks
        for block in self.mix_blocks:
            x = block(x, training=training)

        # Classification head
        x = self.global_pool(x)
        x = self.dropout(x, training=training)
        return self.dense_out(x)

    def get_config(self) -> dict[str, object]:
        config = super().get_config()
        config.update({
            "n_frames": self.n_frames,
            "n_features": self.n_features,
            "filters": self._filters,
        })
        return config


def build_classifier(
    model_type: ModelType = ModelType.inception,
    model_size: ModelSize = ModelSize.small,
    n_frames: int = 99,
    n_features: int = 40,
) -> tf.keras.Model:
    """Factory function to build a classifier from config enums.

    Args:
        model_type: Architecture type (inception or mixednet).
        model_size: Size preset (tiny/small/medium/large).
        n_frames: Number of spectrogram time frames.
        n_features: Number of mel frequency channels.

    Returns:
        Keras model: (batch, n_frames, n_features) → (batch, 1)
    """
    filters, n_blocks = MODEL_SIZE_PRESETS[model_size]

    if model_type == ModelType.inception:
        # For inception: n_blocks is inception block count, add 2 CNN1 blocks
        n_cnn1 = min(2, max(1, n_blocks // 2))
        return InceptionClassifier(
            n_frames=n_frames,
            n_features=n_features,
            filters=filters,
            n_cnn1_blocks=n_cnn1,
            n_inception_blocks=n_blocks,
        )
    elif model_type == ModelType.mixednet:
        return MixedNetClassifier(
            n_frames=n_frames,
            n_features=n_features,
            filters=filters,
            n_blocks=n_blocks,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
