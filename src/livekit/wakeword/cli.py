"""Typer CLI — replaces all notebook functionality with scripted commands."""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.logging import RichHandler

from .config import load_config

app = typer.Typer(
    name="livekit-wakeword",
    help="Wake word detection with microWakeWord-based TFLite models.",
    no_args_is_help=True,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("livekit.wakeword")


@app.command()
def setup(
    data_dir: str = typer.Option("./data", help="Root data directory"),
    skip_negatives: bool = typer.Option(
        False, "--skip-negatives", help="Skip downloading negative speech features"
    ),
) -> None:
    """Download external dependencies: VITS TTS model, negative features, RIRs, backgrounds."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    logger.info("Setting up livekit-wakeword data dependencies...")

    # Download VITS TTS model (.pt checkpoint + config)
    piper_dir = data_path / "piper"
    piper_dir.mkdir(exist_ok=True)
    _download_piper(piper_dir)

    # Download negative speech features (microWakeWord precomputed spectrograms)
    features_dir = data_path / "features"
    features_dir.mkdir(exist_ok=True)
    if skip_negatives:
        logger.info("Skipping negative speech features (--skip-negatives).")
    else:
        _download_microwakeword_features(features_dir)

    # Download RIRs
    rir_dir = data_path / "rirs"
    rir_dir.mkdir(exist_ok=True)
    _download_rirs(rir_dir)

    # Download background noise (MUSAN noise subset)
    bg_dir = data_path / "backgrounds"
    bg_dir.mkdir(exist_ok=True)
    _download_musan_noise(bg_dir)

    logger.info("Setup complete!")


def _download_piper(piper_dir: Path) -> None:
    """Download VITS state_dict and config JSON."""
    import urllib.request

    from rich.progress import Progress

    base_url = "https://github.com/livekit/livekit-wakeword/releases/download/v0.1.0"

    # 1) Download state_dict .pt
    pt_url = f"{base_url}/en-us-libritts-high.state_dict.pt"
    pt_dest = piper_dir / "en-us-libritts-high.pt"
    if not pt_dest.exists():
        logger.info("Downloading en-us-libritts-high.pt (~166 MB)...")
        try:
            with Progress() as progress:
                task = progress.add_task("[cyan]en-us-libritts-high.pt", total=None)
                tmp_path = pt_dest.with_suffix(".tmp")

                def _reporthook(block_num: int, block_size: int, total: int) -> None:
                    if total > 0:
                        progress.update(task, total=total, completed=block_num * block_size)

                urllib.request.urlretrieve(pt_url, str(tmp_path), reporthook=_reporthook)
                tmp_path.rename(pt_dest)
            logger.info("Downloaded en-us-libritts-high.pt")
        except Exception as e:
            logger.warning(f"Failed to download VITS checkpoint: {e}")
            tmp_path = pt_dest.with_suffix(".tmp")
            if tmp_path.exists():
                tmp_path.unlink()
    else:
        logger.info(f"VITS checkpoint already exists: {pt_dest}")

    # 2) Download config JSON (must be next to .pt with .json suffix)
    json_url = f"{base_url}/en-us-libritts-high.config.json"
    json_dest = piper_dir / "en-us-libritts-high.json"
    if not json_dest.exists():
        logger.info("Downloading VITS config JSON...")
        try:
            urllib.request.urlretrieve(json_url, str(json_dest))
            logger.info("Downloaded VITS config JSON")
        except Exception as e:
            logger.warning(f"Failed to download VITS config JSON: {e}")
    else:
        logger.info(f"VITS config already exists: {json_dest}")


def _download_microwakeword_features(features_dir: Path) -> None:
    """Download precomputed spectrogram features from HuggingFace.

    Downloads ZIP files from binhpham/livekit_embedded_wakeword_features,
    extracts them, and converts the RaggedMmap (mmap-ninja) format to .npy files.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.warning(
            "huggingface-hub not installed. Install with: uv pip install huggingface-hub"
        )
        return

    import zipfile

    repo_id = "binhpham/livekit_embedded_wakeword_features"

    # Training negatives get a raw_ prefix so _combine_negative_features
    # can merge them into a single speech_features.npy without self-collision.
    _raw_prefix_zips = {"speech.zip", "no_speech.zip", "speech_background.zip",
                        "no_speech_background.zip"}
    feature_zips = [
        "speech.zip",
        "no_speech.zip",
        "speech_background.zip",
        "no_speech_background.zip",
        "dinner_party.zip",
        "dinner_party_eval.zip",
    ]

    for zip_name in feature_zips:
        base_name = zip_name.replace(".zip", "_features.npy")
        npy_name = f"raw_{base_name}" if zip_name in _raw_prefix_zips else base_name
        npy_path = features_dir / npy_name
        if npy_path.exists():
            logger.info(f"Features already exist: {npy_path}")
            continue

        logger.info(f"Downloading {zip_name} from {repo_id}...")
        try:
            zip_path = hf_hub_download(
                repo_id=repo_id,
                filename=zip_name,
                local_dir=str(features_dir),
                repo_type="dataset",
            )

            # Extract ZIP
            extract_dir = features_dir / zip_name.replace(".zip", "")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(str(extract_dir))

            # Convert RaggedMmap to .npy
            _convert_ragged_to_npy(extract_dir, npy_path)
            logger.info(f"Converted {zip_name} → {npy_path}")

        except Exception as e:
            logger.warning(f"Failed to download/convert {zip_name}: {e}")

    # Create combined feature files for training
    _combine_negative_features(features_dir)


def _find_mmap_dir(extract_dir: Path) -> Path:
    """Find the actual RaggedMmap directory within a nested zip extraction.

    The HuggingFace dataset zips have nested structures like:
      speech/speech/training/voices_lav_clo_training_mmap/
    We need the directory that contains mmap-ninja files (dtype.ninja etc.)
    at its root level and whose name ends with '_mmap'.
    """
    for candidate in extract_dir.rglob("*_mmap"):
        if candidate.is_dir() and (candidate / "dtype.ninja").exists():
            return candidate
    # Fallback: try the extract_dir itself (flat layout)
    return extract_dir


def _convert_ragged_to_npy(ragged_dir: Path, output_path: Path) -> None:
    """Convert mmap-ninja RaggedMmap directory to a fixed-size .npy file.

    Each spectrogram is padded/truncated to 99 frames × 40 features.
    """
    import numpy as np

    from .models.feature_extractor import NUM_CHANNELS, compute_n_frames

    n_frames = compute_n_frames(int(2.0 * 16000))  # 99 for 2.0s

    try:
        from mmap_ninja.ragged import RaggedMmap

        mmap_dir = _find_mmap_dir(ragged_dir)
        logger.info(f"Found mmap directory: {mmap_dir}")
        mmap_data = RaggedMmap(str(mmap_dir))
        n_samples = len(mmap_data)

        if n_samples == 0:
            np.save(str(output_path), np.zeros((0, n_frames, NUM_CHANNELS), dtype=np.float32))
            return

        # Dataset README: "multiply by a factor of 0.0390625 after casting to a float"
        uint16_to_float_scale = 0.0390625

        all_specs: list[np.ndarray] = []
        for i in range(n_samples):
            spec = np.array(mmap_data[i], dtype=np.float32) * uint16_to_float_scale
            # Pad or truncate to canonical n_frames
            if spec.shape[0] >= n_frames:
                spec = spec[-n_frames:]
            else:
                pad = np.zeros((n_frames - spec.shape[0], spec.shape[1]), dtype=np.float32)
                spec = np.concatenate([pad, spec], axis=0)
            # Ensure correct feature count
            if spec.shape[1] != NUM_CHANNELS:
                logger.warning(
                    f"Feature count mismatch: expected {NUM_CHANNELS}, got {spec.shape[1]}"
                )
                continue
            all_specs.append(spec)

        features = np.stack(all_specs, axis=0)
        np.save(str(output_path), features)
        logger.info(f"Saved {features.shape} features to {output_path}")

    except ImportError:
        logger.warning(
            "mmap-ninja not installed. Install with: uv pip install mmap-ninja\n"
            "Cannot convert RaggedMmap features to .npy format."
        )


def _combine_negative_features(features_dir: Path) -> None:
    """Combine individual negative feature files into training/validation sets.

    Individual source files are saved with a ``_raw_`` prefix to avoid
    name-collision with the combined output (``speech_features.npy``).
    """
    import numpy as np

    # Training negatives: individual raw files → combined speech_features.npy
    raw_files = [
        "raw_speech_features.npy",
        "raw_no_speech_features.npy",
        "raw_speech_background_features.npy",
        "raw_no_speech_background_features.npy",
    ]
    train_arrays = []
    for fname in raw_files:
        fpath = features_dir / fname
        if fpath.exists():
            arr = np.load(str(fpath))
            if arr.shape[0] > 0:
                train_arrays.append(arr)

    if train_arrays:
        combined = np.concatenate(train_arrays, axis=0)
        out_path = features_dir / "speech_features.npy"
        np.save(str(out_path), combined)
        logger.info(f"Combined negative training features: {combined.shape}")

    # Validation negatives: dinner_party_eval
    eval_path = features_dir / "dinner_party_eval_features.npy"
    if eval_path.exists():
        val_path = features_dir / "validation_features.npy"
        import shutil

        shutil.copy2(str(eval_path), str(val_path))
        logger.info(f"Validation features: {val_path}")


def _download_musan_noise(bg_dir: Path) -> None:
    """Download MUSAN noise subset (~6 hrs of background noise)."""
    # Check if noise files already present
    existing = list(bg_dir.glob("**/*.wav"))
    if existing:
        logger.info(f"Background noise already present: {len(existing)} files in {bg_dir}")
        return

    try:
        from huggingface_hub import snapshot_download

        logger.info("Downloading MUSAN background audio from HuggingFace (~1.1 GB)...")
        snapshot_download(
            repo_id="FluidInference/musan",
            repo_type="dataset",
            allow_patterns="**/*.wav",
            local_dir=str(bg_dir),
        )
        extracted = list(bg_dir.glob("**/*.wav"))
        logger.info(f"Downloaded {len(extracted)} background noise files")
    except ImportError:
        logger.warning(
            "huggingface-hub not installed. Install with: uv pip install huggingface-hub"
        )
    except Exception as e:
        logger.warning(f"Failed to download MUSAN noise: {e}")


def _download_rirs(rir_dir: Path) -> None:
    """Download MIT room impulse responses (~8 MB, 270 WAV files)."""
    try:
        from huggingface_hub import snapshot_download

        logger.info("Downloading MIT room impulse responses (~8 MB)...")
        snapshot_download(
            repo_id="davidscripka/MIT_environmental_impulse_responses",
            repo_type="dataset",
            allow_patterns="16khz/*.wav",
            local_dir=str(rir_dir),
        )
        logger.info("RIRs downloaded.")
    except ImportError:
        logger.warning(
            "huggingface-hub not installed. Install with: uv pip install huggingface-hub"
        )
    except Exception as e:
        logger.warning(f"Failed to download RIRs: {e}")


@app.command()
def generate(
    config_path: str = typer.Argument(..., help="Path to wake word config YAML"),
) -> None:
    """Generate synthetic speech clips (positive + adversarial negative)."""
    config = load_config(config_path)

    logger.info(f"Generating data for '{config.model_name}'...")
    logger.info(f"Target phrases: {config.target_phrases}")

    from .data.generate import run_generate

    run_generate(config)
    logger.info("Generation complete!")


@app.command()
def augment(
    config_path: str = typer.Argument(..., help="Path to wake word config YAML"),
) -> None:
    """Augment clips and extract spectrogram features."""
    config = load_config(config_path)

    logger.info(f"Augmenting data for '{config.model_name}'...")

    from .data.augment import run_augment
    from .data.features import run_extraction

    run_augment(config)
    logger.info("Augmentation complete!")

    logger.info("Extracting spectrogram features...")
    run_extraction(config)
    logger.info("Feature extraction complete!")


@app.command()
def train(
    config_path: str = typer.Argument(..., help="Path to wake word config YAML"),
) -> None:
    """Train classifier on extracted features (3-phase adaptive training)."""
    config = load_config(config_path)

    from .utils import configure_gpu

    configure_gpu()

    logger.info(f"Training '{config.model_name}' model...")
    logger.info(f"Model: {config.model.model_type.value} ({config.model.model_size.value})")
    logger.info(f"Steps: {config.steps}")

    from .training.trainer import run_train

    model_path = run_train(config)
    logger.info(f"Training complete! Model saved to {model_path}")


@app.command()
def export(
    config_path: str = typer.Argument(..., help="Path to wake word config YAML"),
    quantize: bool = typer.Option(False, "--quantize", help="Apply INT8 quantization"),
) -> None:
    """Export trained model to TFLite (optionally quantize for embedded)."""
    config = load_config(config_path)

    logger.info(f"Exporting '{config.model_name}' to TFLite...")

    from .export.tflite import run_export

    tflite_path = run_export(config, quantize=quantize)
    logger.info(f"Export complete! TFLite model at {tflite_path}")


@app.command()
def eval(
    config_path: str = typer.Argument(..., help="Path to wake word config YAML"),
    model_path: str = typer.Option(
        None,
        "--model",
        "-m",
        help="Path to TFLite model (default: <output_dir>/<model_name>.tflite)",
    ),
) -> None:
    """Evaluate model on validation set: DET curve, AUT, FPPH, recall."""
    from pathlib import Path

    config = load_config(config_path)

    if model_path is None:
        resolved_model = config.model_output_dir / f"{config.model_name}.tflite"
    else:
        resolved_model = Path(model_path)

    logger.info(f"Evaluating '{config.model_name}' with model {resolved_model}...")

    from .eval.evaluate import run_eval

    results = run_eval(config, resolved_model)

    logger.info(
        f"AUT={results['aut']:.4f}  FPPH={results['fpph']:.2f}  "
        f"Recall={results['recall']:.1%}  Threshold={results['threshold']:.2f}"
    )
    logger.info(f"DET curve: {config.model_output_dir / f'{config.model_name}_det.png'}")


@app.command()
def run(
    config_path: str = typer.Argument(..., help="Path to wake word config YAML"),
) -> None:
    """Run entire pipeline end-to-end: generate → augment → train → export."""
    config = load_config(config_path)

    logger.info(f"Running full pipeline for '{config.model_name}'...")

    # Configure GPU before any TF imports to ensure memory growth is set
    from .utils import configure_gpu

    configure_gpu()

    from .data.augment import run_augment
    from .data.features import run_extraction
    from .data.generate import run_generate
    from .eval.evaluate import run_eval
    from .export.tflite import run_export
    from .training.trainer import run_train

    logger.info("Step 1/6: Generate synthetic data")
    run_generate(config)

    logger.info("Step 2/6: Augment clips")
    run_augment(config)

    logger.info("Step 3/6: Extract spectrogram features")
    run_extraction(config)

    logger.info("Step 4/6: Train classifier")
    run_train(config)

    logger.info("Step 5/6: Export to TFLite")
    tflite_path = run_export(config)

    logger.info("Step 6/6: Evaluate model")
    results = run_eval(config, tflite_path)
    logger.info(
        f"Eval: AUT={results['aut']:.4f}  FPPH={results['fpph']:.2f}  "
        f"Recall={results['recall']:.1%}"
    )

    logger.info("Full pipeline complete!")
