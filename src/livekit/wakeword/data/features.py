"""Feature extraction: audio → micro_speech spectrogram → .npy."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..config import WakeWordConfig
from ..models.feature_extractor import MicroFrontend, compute_n_frames

logger = logging.getLogger(__name__)


def extract_features_from_directory(
    clip_dir: Path,
    frontend: MicroFrontend,
    n_frames: int,
    batch_size: int = 32,
) -> np.ndarray:
    """Extract (N_clips, n_frames, 40) features from a directory of WAV files.

    Processes clips through MicroFrontend, then pads/truncates to
    canonical n_frames per clip.
    """
    import soundfile as sf
    from tqdm import tqdm

    wav_files = sorted(clip_dir.glob("*.wav"))
    if not wav_files:
        logger.warning(f"No WAV files in {clip_dir}")
        return np.zeros((0, n_frames, frontend.num_channels), dtype=np.float32)

    all_features: list[np.ndarray] = []

    for wav_path in tqdm(wav_files, desc=f"Features {clip_dir.name}", unit="clip"):
        audio, sr = sf.read(str(wav_path))
        if audio.ndim > 1:
            audio = audio[:, 0]
        audio = audio.astype(np.float32)

        # Compute spectrogram: (1, actual_frames, num_channels)
        spec = frontend(audio)
        clip_spec = spec[0]  # (actual_frames, num_channels)

        # Pad or truncate to canonical n_frames
        if clip_spec.shape[0] >= n_frames:
            clip_spec = clip_spec[-n_frames:]
        else:
            pad = np.zeros(
                (n_frames - clip_spec.shape[0], frontend.num_channels),
                dtype=np.float32,
            )
            clip_spec = np.concatenate([pad, clip_spec], axis=0)

        all_features.append(clip_spec)

    if not all_features:
        return np.zeros((0, n_frames, frontend.num_channels), dtype=np.float32)

    return np.stack(all_features, axis=0)  # (N_clips, n_frames, num_channels)


def run_extraction(config: WakeWordConfig) -> None:
    """Extract and save features for all splits of a wake word config."""
    frontend = MicroFrontend(
        window_size_ms=config.augmentation.window_size_ms,
        window_step_ms=config.augmentation.window_step_ms,
        num_channels=config.augmentation.num_channels,
    )
    n_frames = config.n_frames

    model_dir = config.model_output_dir
    splits = [
        ("positive_train", "positive_features_train.npy"),
        ("positive_test", "positive_features_test.npy"),
        ("negative_train", "negative_features_train.npy"),
        ("negative_test", "negative_features_test.npy"),
    ]

    for clip_subdir, feature_filename in splits:
        clip_dir = model_dir / clip_subdir
        if not clip_dir.exists():
            logger.warning(f"Skipping feature extraction for {clip_subdir}: not found")
            continue

        logger.info(f"Extracting features from {clip_dir}...")
        features = extract_features_from_directory(
            clip_dir=clip_dir,
            frontend=frontend,
            n_frames=n_frames,
            batch_size=config.augmentation.batch_size,
        )

        out_path = model_dir / feature_filename
        np.save(str(out_path), features)
        logger.info(f"Saved {features.shape} features to {out_path}")
