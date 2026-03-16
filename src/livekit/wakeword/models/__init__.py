"""Neural network models for wake word detection."""

# Feature extractor uses only numpy — safe to import eagerly.
from .feature_extractor import MicroFrontend

# Classifier / pipeline modules require tensorflow and are lazy-loaded so that
# the inference-only path (numpy + tflite-runtime) works without tensorflow.
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "InceptionClassifier": (".classifier", "InceptionClassifier"),
    "MixedNetClassifier": (".classifier", "MixedNetClassifier"),
    "SubSpectralNormalization": (".classifier", "SubSpectralNormalization"),
    "build_classifier": (".classifier", "build_classifier"),
    "WakeWordClassifier": (".pipeline", "WakeWordClassifier"),
}


def __getattr__(name: str) -> object:
    if name in _LAZY_IMPORTS:
        import importlib

        module_path, attr = _LAZY_IMPORTS[name]
        mod = importlib.import_module(module_path, __name__)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "InceptionClassifier",
    "MicroFrontend",
    "MixedNetClassifier",
    "SubSpectralNormalization",
    "WakeWordClassifier",
    "build_classifier",
]
