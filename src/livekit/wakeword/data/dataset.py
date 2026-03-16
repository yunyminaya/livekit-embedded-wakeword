"""Memory-mapped dataset for training with mixed positive/negative batches."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

logger = logging.getLogger(__name__)


def mmap_batch_generator(
    data_files: dict[str, str | Path],
    n_per_class: dict[str, int],
    label_funcs: dict[str, Callable[[np.ndarray], int]],
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Generate mixed batches from memory-mapped .npy files.

    Each batch contains samples from each class according to n_per_class.
    Files are memory-mapped so data larger than RAM can be used.

    Yields:
        (features, labels) where features is (batch_size, n_frames, num_channels)
        and labels is (batch_size,) with 0/1 values.
    """
    # Memory-map all files
    mmaps: dict[str, np.ndarray] = {}
    for name, path in data_files.items():
        path = Path(path)
        if not path.exists():
            logger.warning(f"Data file not found: {path}, skipping class '{name}'")
            continue
        data = np.load(str(path), mmap_mode="r")
        # Validate: expect 3D (N, n_frames, num_channels)
        if data.ndim != 3:
            raise ValueError(
                f"Expected 3D features for '{name}', got shape {data.shape}. "
                f"The file {path} may need to be re-extracted."
            )
        mmaps[name] = data
        logger.info(f"Loaded {name}: shape={mmaps[name].shape} from {path}")

    if not mmaps:
        raise FileNotFoundError("No data files found for training")

    # Warn about requested classes that have no loaded data
    for name, n in n_per_class.items():
        if n > 0 and name not in mmaps:
            logger.warning(
                f"Class '{name}' requested {n} samples per batch but no data file was loaded"
            )

    # Track position in each file
    positions: dict[str, int] = {name: 0 for name in mmaps}

    while True:
        batch_features: list[np.ndarray] = []
        batch_labels: list[int] = []

        for name, data in mmaps.items():
            n = n_per_class.get(name, 0)
            if n == 0:
                continue

            label_fn = label_funcs[name]
            total = data.shape[0]
            pos = positions[name]

            # Collect n samples, wrapping around if needed
            indices = np.arange(pos, pos + n) % total
            samples = data[indices]

            for sample in samples:
                batch_features.append(sample)
                batch_labels.append(label_fn(sample))

            positions[name] = (pos + n) % total

        if not batch_features:
            break

        features = np.stack(batch_features, axis=0)
        labels = np.array(batch_labels, dtype=np.float32)

        # Shuffle within batch
        perm = np.random.permutation(len(labels))
        yield features[perm], labels[perm]


def create_tf_dataset(
    data_files: dict[str, str | Path],
    n_per_class: dict[str, int],
    label_funcs: dict[str, Callable[[np.ndarray], int]],
    n_frames: int = 99,
    num_channels: int = 40,
) -> object:
    """Create a tf.data.Dataset from the memory-mapped batch generator.

    Returns a tf.data.Dataset yielding (features, labels) batches.
    """
    import tensorflow as tf

    def _gen() -> Iterator[tuple[np.ndarray, np.ndarray]]:
        yield from mmap_batch_generator(data_files, n_per_class, label_funcs)

    return tf.data.Dataset.from_generator(
        _gen,
        output_signature=(
            tf.TensorSpec(shape=(None, n_frames, num_channels), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
        ),
    ).prefetch(tf.data.AUTOTUNE)
