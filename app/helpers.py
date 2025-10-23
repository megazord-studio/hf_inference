"""Backward-compatible helpers module.

This module re-exports functions from the new infrastructure layer
to maintain backward compatibility with existing code.

Deprecated: Import directly from app.infrastructure modules instead.
"""

# Re-export from infrastructure modules for backward compatibility
from app.infrastructure.data import to_dataframe
from app.infrastructure.device import device_arg
from app.infrastructure.device import device_str
from app.infrastructure.file_io import audio_to_bytes
from app.infrastructure.file_io import ensure_image
from app.infrastructure.file_io import get_upload_file_image
from app.infrastructure.file_io import get_upload_file_path
from app.infrastructure.file_io import image_to_bytes
from app.infrastructure.response import safe_json
from app.infrastructure.response import safe_print_output

__all__ = [
    # Device utilities
    "device_str",
    "device_arg",
    # Response formatting
    "safe_json",
    "safe_print_output",
    # File I/O
    "ensure_image",
    "get_upload_file_image",
    "get_upload_file_path",
    "image_to_bytes",
    "audio_to_bytes",
    # Data conversion
    "to_dataframe",
]
