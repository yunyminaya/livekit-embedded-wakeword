"""Bundled resources for livekit-wakeword.

The microWakeWord architecture uses the TFLite Micro microfrontend
C library (via pymicro-features) for feature extraction, so no
bundled model files are required.
"""

from importlib import resources
from pathlib import Path


def get_resource_path(filename: str) -> Path:
    """Get the path to a bundled resource file."""
    return Path(str(resources.files(__package__) / filename))
