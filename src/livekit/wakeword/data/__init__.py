"""Data generation, augmentation, and dataset utilities."""

from .augment import AudioAugmentor, align_clip_to_end, run_augment
from .dataset import create_tf_dataset, mmap_batch_generator
from .features import extract_features_from_directory, run_extraction

# Generate module requires torch (datagen extra) — lazy-load to avoid
# ImportError when torch is not installed.
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "generate_adversarial_phrases": (".generate", "generate_adversarial_phrases"),
    "run_generate": (".generate", "run_generate"),
    "synthesize_clips": (".generate", "synthesize_clips"),
}


def __getattr__(name: str) -> object:
    if name in _LAZY_IMPORTS:
        import importlib

        module_path, attr = _LAZY_IMPORTS[name]
        mod = importlib.import_module(module_path, __name__)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AudioAugmentor",
    "align_clip_to_end",
    "create_tf_dataset",
    "extract_features_from_directory",
    "generate_adversarial_phrases",
    "mmap_batch_generator",
    "run_augment",
    "run_extraction",
    "run_generate",
    "synthesize_clips",
]
