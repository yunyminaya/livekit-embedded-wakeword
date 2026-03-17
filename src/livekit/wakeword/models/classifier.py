"""TensorFlow/Keras classification heads for wake word detection.

Implements Inception and MixedNet CNN architectures inspired by microWakeWord.
Input: (batch, N_frames, N_features) log-mel spectrogram features.
Output: (batch, 1) confidence score in [0, 1].
"""

from __future__ import annotations

import numpy as np
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

    def call(self, x: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        # x: (batch, time, freq, channels)
        sub_bands = tf.split(x, self.num_sub_bands, axis=2)
        normalized = [bn(sb, training=training) for bn, sb in zip(self._batch_norms, sub_bands)]
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

    def call(self, x: tf.Tensor, training: bool | None = None) -> tf.Tensor:
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
        config.update(
            {
                "filters": self.filters,
                "kernel_time": self.kernel_time,
                "kernel_freq": self.kernel_freq,
                "dilation": self.dilation,
            }
        )
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
        self.cnn1_blocks: list[
            tuple[
                tf.keras.layers.Conv2D,
                SubSpectralNormalization,
                tf.keras.layers.ReLU,
            ]
        ] = []
        for i in range(n_cnn1_blocks):
            conv = tf.keras.layers.Conv2D(filters, (5, 5), padding="same", name=f"cnn1_conv_{i}")
            ssn = SubSpectralNormalization(num_sub_bands=5, name=f"cnn1_ssn_{i}")
            relu = tf.keras.layers.ReLU(name=f"cnn1_relu_{i}")
            self.cnn1_blocks.append((conv, ssn, relu))

        # Inception blocks
        self.inception_blocks: list[InceptionBlock] = []
        for i in range(n_inception_blocks):
            dilation = 1 + (i % 3)  # cycle through 1, 2, 3
            block = InceptionBlock(filters=filters, dilation=dilation, name=f"inception_{i}")
            self.inception_blocks.append(block)

        # Classification head
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.dense_out = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, x: tf.Tensor, training: bool | None = None) -> tf.Tensor:
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
        config.update(
            {
                "n_frames": self.n_frames,
                "n_features": self.n_features,
                "filters": self._filters,
            }
        )
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
            self.residual_proj = tf.keras.layers.Conv2D(self.filters, (1, 1), padding="same")

        # Optional channel attention (CBAM-style squeeze-and-excitation)
        if self.use_attention:
            self.se_pool = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)
            self.se_fc1 = tf.keras.layers.Dense(max(1, self.filters // 4), activation="relu")
            self.se_fc2 = tf.keras.layers.Dense(self.filters, activation="sigmoid")

        super().build(input_shape)

    def call(self, x: tf.Tensor, training: bool | None = None) -> tf.Tensor:
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
        config.update(
            {
                "filters": self.filters,
                "num_groups": self.num_groups,
                "use_attention": self.use_attention,
            }
        )
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
        self.init_conv = tf.keras.layers.Conv2D(filters, (3, 3), padding="same", name="init_conv")
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

    def call(self, x: tf.Tensor, training: bool | None = None) -> tf.Tensor:
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
        config.update(
            {
                "n_frames": self.n_frames,
                "n_features": self.n_features,
                "filters": self._filters,
            }
        )
        return config


def _fuse_conv_bn(
    conv: tf.keras.layers.Conv2D,
    bn: tf.keras.layers.BatchNormalization,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse Conv2D (no bias) + BatchNormalization into a single (kernel, bias).

    Returns:
        (kernel_fused, bias_fused) numpy arrays.
    """
    kernel = conv.kernel.numpy()
    gamma = bn.gamma.numpy()
    beta = bn.beta.numpy()
    mean = bn.moving_mean.numpy()
    var = bn.moving_variance.numpy()
    eps = bn.epsilon

    scale = gamma / np.sqrt(var + eps)
    kernel_fused = kernel * scale[np.newaxis, np.newaxis, np.newaxis, :]
    bias_fused = -mean * scale + beta
    return kernel_fused, bias_fused


def _fuse_dw_conv_bn(
    dw_conv: tf.keras.layers.DepthwiseConv2D,
    bn: tf.keras.layers.BatchNormalization,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse DepthwiseConv2D (no bias, depth_multiplier=1) + BatchNormalization.

    For depthwise conv the kernel shape is (H, W, C_in, 1) and BN scale
    is applied along the channel-in axis (axis 2).

    Returns:
        (kernel_fused, bias_fused) numpy arrays.
    """
    kernel = dw_conv.kernel.numpy()  # (H, W, C, 1)
    gamma = bn.gamma.numpy()
    beta = bn.beta.numpy()
    mean = bn.moving_mean.numpy()
    var = bn.moving_variance.numpy()
    eps = bn.epsilon

    scale = gamma / np.sqrt(var + eps)
    kernel_fused = kernel * scale[np.newaxis, np.newaxis, :, np.newaxis]
    bias_fused = -mean * scale + beta
    return kernel_fused, bias_fused


def _bn_to_scale_bias(
    bn: tf.keras.layers.BatchNormalization,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract fused scale + bias from a standalone BatchNormalization (identity branch)."""
    gamma = bn.gamma.numpy()
    beta = bn.beta.numpy()
    mean = bn.moving_mean.numpy()
    var = bn.moving_variance.numpy()
    eps = bn.epsilon
    scale = gamma / np.sqrt(var + eps)
    bias = -mean * scale + beta
    return scale, bias


class RepDSBlock(tf.keras.layers.Layer):  # type: ignore[misc]
    """RepVGG-style depthwise-separable block with temporal dilation.

    Training-time multi-branch depthwise convolution:
      A: DepthwiseConv2D(3x3, dilation=(d,1)) + BN
      B: DepthwiseConv2D(1x1) + BN
      C: BN only (identity shortcut)
    Summed → ReLU → Pointwise Conv2D(1x1) + BN → ReLU

    Residual: input added to output when channel counts match.

    At export, the depthwise branches fuse into a single dilated DW-Conv + bias,
    and the pointwise conv+BN fuses into a single PW-Conv + bias. No BN remains.
    ESP32-S3 SIMD handles the fused DW+PW conv pair efficiently.
    """

    def __init__(self, filters: int, dilation: int = 1, **kwargs: object):
        super().__init__(**kwargs)
        self._filters = filters
        self._dilation = dilation

    def build(self, input_shape: tf.TensorShape) -> None:
        in_channels = int(input_shape[-1])
        d = self._dilation

        # DW Branch A: 3x3 dilated (temporal dilation only)
        self.dw_conv3x3 = tf.keras.layers.DepthwiseConv2D(
            (3, 3),
            dilation_rate=(d, 1),
            padding="same",
            use_bias=False,
            depth_multiplier=1,
            name="dw_conv3x3",
        )
        self.dw_bn3x3 = tf.keras.layers.BatchNormalization(name="dw_bn3x3")

        # DW Branch B: 1x1
        self.dw_conv1x1 = tf.keras.layers.DepthwiseConv2D(
            (1, 1),
            padding="same",
            use_bias=False,
            depth_multiplier=1,
            name="dw_conv1x1",
        )
        self.dw_bn1x1 = tf.keras.layers.BatchNormalization(name="dw_bn1x1")

        # DW Branch C: identity (always valid for depthwise since channels unchanged)
        self.dw_bn_id = tf.keras.layers.BatchNormalization(name="dw_bn_id")

        self.dw_relu = tf.keras.layers.ReLU(name="dw_relu")

        # Pointwise: channel mixing
        self.pw_conv = tf.keras.layers.Conv2D(
            self._filters, (1, 1), padding="same", use_bias=False, name="pw_conv"
        )
        self.pw_bn = tf.keras.layers.BatchNormalization(name="pw_bn")
        self.pw_relu = tf.keras.layers.ReLU(name="pw_relu")

        # Residual when input channels == output filters
        self._has_residual = in_channels == self._filters

        super().build(input_shape)

    def call(self, x: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        # Multi-branch depthwise
        a = self.dw_bn3x3(self.dw_conv3x3(x), training=training)
        b = self.dw_bn1x1(self.dw_conv1x1(x), training=training)
        c = self.dw_bn_id(x, training=training)
        dw_out = self.dw_relu(a + b + c)

        # Pointwise
        pw_out = self.pw_relu(self.pw_bn(self.pw_conv(dw_out), training=training))

        # Residual
        if self._has_residual:
            return pw_out + x
        return pw_out

    def fuse(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fuse all branches into (dw_kernel, dw_bias, pw_kernel, pw_bias).

        The depthwise 3x3 dilated + 1x1 + identity branches merge into one
        3x3 depthwise kernel (applied with the original dilation at runtime).
        The pointwise conv+BN merges into a single 1x1 conv + bias.
        """
        # Branch A: 3x3 depthwise (already correct shape)
        k3, b3 = _fuse_dw_conv_bn(self.dw_conv3x3, self.dw_bn3x3)  # (3, 3, C, 1)

        # Branch B: 1x1 depthwise → pad to 3x3
        k1, b1 = _fuse_dw_conv_bn(self.dw_conv1x1, self.dw_bn1x1)  # (1, 1, C, 1)
        k1_padded = np.zeros_like(k3)
        k1_padded[1:2, 1:2, :, :] = k1

        # Branch C: identity BN → depthwise delta kernel at center
        scale_id, b_id = _bn_to_scale_bias(self.dw_bn_id)
        k_id = np.zeros_like(k3)
        k_id[1, 1, :, 0] = scale_id

        dw_k = k3 + k1_padded + k_id
        dw_b = b3 + b1 + b_id

        # Pointwise conv + BN
        pw_k, pw_b = _fuse_conv_bn(self.pw_conv, self.pw_bn)

        return dw_k, dw_b, pw_k, pw_b

    def get_config(self) -> dict[str, object]:
        config = super().get_config()
        config["filters"] = self._filters
        config["dilation"] = self._dilation
        return config


class RepCNNClassifier(tf.keras.Model):  # type: ignore[misc]
    """RepVGG-style depthwise-separable wake word classifier.

    Training: multi-branch DW convs (3x3 dilated + 1x1 + identity) + pointwise.
    Deploy: fused DW-Conv + PW-Conv per block via reparameterize_model().

    Temporal dilations cycle [1, 2, 4] across blocks for multi-scale context.

    Architecture:
        Input (N_frames, N_features) → expand_dims → (N_frames, N_features, 1)
        → Conv2D(filters, 3x3) + BN + ReLU  (channel expansion)
        → N × RepDSBlock(filters, dilation=cycling)
        → GlobalAveragePooling2D → Dropout → Dense(1, sigmoid)
    """

    DILATION_CYCLE: tuple[int, ...] = (1, 2, 4)

    def __init__(
        self,
        n_frames: int = 99,
        n_features: int = 40,
        filters: int = 16,
        n_blocks: int = 3,
        dropout: float = 0.1,
        **kwargs: object,
    ):
        super().__init__(**kwargs)
        self.n_frames = n_frames
        self.n_features = n_features
        self._filters = filters

        # Initial conv to expand channels (1 → filters)
        self.init_conv = tf.keras.layers.Conv2D(
            filters, (3, 3), padding="same", use_bias=False, name="init_conv"
        )
        self.init_bn = tf.keras.layers.BatchNormalization(name="init_bn")
        self.init_relu = tf.keras.layers.ReLU(name="init_relu")

        # RepDS blocks with cycling temporal dilation
        self.rep_blocks: list[RepDSBlock] = []
        for i in range(n_blocks):
            d = self.DILATION_CYCLE[i % len(self.DILATION_CYCLE)]
            self.rep_blocks.append(RepDSBlock(filters=filters, dilation=d, name=f"repdsblock_{i}"))

        # Classification head
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.dense_out = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, x: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        x = tf.expand_dims(x, axis=-1)
        x = self.init_relu(self.init_bn(self.init_conv(x), training=training))
        for block in self.rep_blocks:
            x = block(x, training=training)
        x = self.global_pool(x)
        x = self.dropout(x, training=training)
        return self.dense_out(x)

    def get_config(self) -> dict[str, object]:
        config = super().get_config()
        config.update(
            {
                "n_frames": self.n_frames,
                "n_features": self.n_features,
                "filters": self._filters,
            }
        )
        return config


class LiGLUBlock(tf.keras.layers.Layer):  # type: ignore[misc]
    """Linear Gated Linear Unit block with causal dilated depthwise convolution.

    Gate projection → GLU gating → causal depthwise conv → LayerNorm → Swish → output projection.
    Block-level residual connection.
    """

    def __init__(self, channels: int, dilation_rate: int = 1, **kwargs: object):
        super().__init__(**kwargs)
        self._channels = channels
        self._dilation_rate = dilation_rate

    def build(self, input_shape: tf.TensorShape) -> None:
        ch = self._channels

        # Phase 1: Gate projection (pointwise expand to 2x for GLU)
        self.gate_proj = tf.keras.layers.Conv1D(ch * 2, 1, name="gate_proj")

        # Phase 3: Causal depthwise conv
        self.dw_conv = tf.keras.layers.Conv1D(
            ch,
            kernel_size=5,
            groups=ch,
            dilation_rate=self._dilation_rate,
            padding="valid",
            name="dw_conv",
        )

        # Phase 4: LayerNorm
        self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1, name="ln")

        # Phase 6: Output projection (pointwise)
        self.out_proj = tf.keras.layers.Conv1D(ch, 1, name="out_proj")

        # Residual projection if needed
        in_channels = int(input_shape[-1])
        self._needs_projection = in_channels != ch
        if self._needs_projection:
            self.residual_proj = tf.keras.layers.Conv1D(ch, 1, name="residual_proj")

        super().build(input_shape)

    def call(self, x: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        residual = x

        # Phase 1: Gate projection
        projected = self.gate_proj(x)

        # Phase 2: GLU gating — split + value * sigmoid(gate)
        value, gate = tf.split(projected, 2, axis=-1)
        x = value * tf.sigmoid(gate)

        # Phase 3: Causal depthwise conv with manual left-padding
        # For kernel_size=5 and dilation_rate=d, receptive field = (5-1)*d = 4*d
        pad_len = (5 - 1) * self._dilation_rate
        x = tf.pad(x, [[0, 0], [pad_len, 0], [0, 0]])
        x = self.dw_conv(x)

        # Phase 4: LayerNorm
        x = self.layer_norm(x)

        # Phase 5: Swish activation
        x = tf.nn.silu(x)

        # Phase 6: Output projection
        x = self.out_proj(x)

        # Block-level residual
        if self._needs_projection:
            residual = self.residual_proj(residual)
        return x + residual

    def get_config(self) -> dict[str, object]:
        config = super().get_config()
        config["channels"] = self._channels
        config["dilation_rate"] = self._dilation_rate
        return config


class WakeNet9Classifier(tf.keras.Model):  # type: ignore[misc]
    """WakeNet9-style 1D CNN wake word classifier with LiGLU blocks.

    Architecture:
        Input (N_frames, N_features) → Conv1D(filters, 1) + Swish + LayerNorm
        → N × LiGLUBlock (cyclic dilations)
        → Long-range residual (penultimate → last block output, when n_blocks >= 3)
        → GlobalAveragePooling1D → Dropout → Dense(1, sigmoid)
    """

    DILATION_PATTERN: tuple[int, ...] = (1, 2, 3, 4, 3)

    def __init__(
        self,
        n_frames: int = 99,
        n_features: int = 40,
        filters: int = 16,
        n_blocks: int = 3,
        dropout: float = 0.1,
        **kwargs: object,
    ):
        super().__init__(**kwargs)
        self.n_frames = n_frames
        self.n_features = n_features
        self._filters = filters
        self._n_blocks = n_blocks

        # Initial projection: n_features → filters
        self.init_conv = tf.keras.layers.Conv1D(filters, 1, name="init_conv")
        self.init_ln = tf.keras.layers.LayerNormalization(axis=-1, name="init_ln")

        # LiGLU blocks with cyclic dilations
        self.liglu_blocks: list[LiGLUBlock] = []
        for i in range(n_blocks):
            d = self.DILATION_PATTERN[i % len(self.DILATION_PATTERN)]
            self.liglu_blocks.append(
                LiGLUBlock(channels=filters, dilation_rate=d, name=f"liglu_{i}")
            )

        self._use_long_residual = n_blocks >= 3

        # Classification head
        self.global_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.dense_out = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, x: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        """Forward pass.

        Args:
            x: (batch, N_frames, N_features) spectrogram features

        Returns:
            (batch, 1) confidence score [0, 1]
        """
        # Input is already (batch, time, channels) — suitable for Conv1D
        x = self.init_conv(x)
        x = tf.nn.silu(x)
        x = self.init_ln(x)

        # LiGLU blocks with long-range residual
        for i, block in enumerate(self.liglu_blocks):
            # Save penultimate block output for long-range residual
            if self._use_long_residual and i == len(self.liglu_blocks) - 2:
                long_residual = x
            x = block(x, training=training)

        # Add long-range residual after last block
        if self._use_long_residual:
            x = x + long_residual

        # Classification head
        x = self.global_pool(x)
        x = self.dropout(x, training=training)
        return self.dense_out(x)

    def get_config(self) -> dict[str, object]:
        config = super().get_config()
        config.update(
            {
                "n_frames": self.n_frames,
                "n_features": self.n_features,
                "filters": self._filters,
            }
        )
        return config


def reparameterize_model(rep_model: RepCNNClassifier) -> tf.keras.Model:
    """Fuse RepCNN multi-branch blocks into plain DW-Conv + PW-Conv layers.

    Creates a functional Keras model with no BatchNormalization layers,
    suitable for TFLite export and quantization. Each RepDSBlock becomes
    a single dilated DepthwiseConv2D + Conv2D(1x1) pair.

    Args:
        rep_model: A built RepCNNClassifier (must have been called at least once).

    Returns:
        A plain tf.keras.Model with fused weights and identical output.
    """
    n_frames = rep_model.n_frames
    n_features = rep_model.n_features

    inp = tf.keras.Input(shape=(n_frames, n_features))
    x = tf.keras.layers.Reshape((n_frames, n_features, 1))(inp)

    # Fuse init conv + BN
    k_init, b_init = _fuse_conv_bn(rep_model.init_conv, rep_model.init_bn)
    init_conv = tf.keras.layers.Conv2D(
        k_init.shape[-1], (3, 3), padding="same", name="fused_init_conv"
    )
    x = init_conv(x)
    init_conv.set_weights([k_init, b_init])
    x = tf.keras.layers.ReLU()(x)

    # Fuse each RepDSBlock → DW-Conv + PW-Conv
    for i, block in enumerate(rep_model.rep_blocks):
        dw_k, dw_b, pw_k, pw_b = block.fuse()
        residual = x

        # Fused dilated depthwise conv
        dw = tf.keras.layers.DepthwiseConv2D(
            (3, 3),
            dilation_rate=(block._dilation, 1),
            padding="same",
            depth_multiplier=1,
            name=f"fused_dw_{i}",
        )
        x = dw(x)
        dw.set_weights([dw_k, dw_b])
        x = tf.keras.layers.ReLU()(x)

        # Fused pointwise conv
        pw = tf.keras.layers.Conv2D(pw_k.shape[-1], (1, 1), padding="same", name=f"fused_pw_{i}")
        x = pw(x)
        pw.set_weights([pw_k, pw_b])
        x = tf.keras.layers.ReLU()(x)

        # Residual
        if block._has_residual:
            x = x + residual

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    dense = tf.keras.layers.Dense(1, activation="sigmoid", name="fused_dense")
    x = dense(x)
    dense.set_weights(rep_model.dense_out.get_weights())

    return tf.keras.Model(inputs=inp, outputs=x, name="fused_repcnn")


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
    elif model_type == ModelType.repcnn:
        return RepCNNClassifier(
            n_frames=n_frames,
            n_features=n_features,
            filters=filters,
            n_blocks=n_blocks,
        )
    elif model_type == ModelType.wakenet9:
        return WakeNet9Classifier(
            n_frames=n_frames,
            n_features=n_features,
            filters=filters,
            n_blocks=n_blocks,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
