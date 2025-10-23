"""File I/O utilities for handling multipart uploads and conversions.

Provides clean abstractions for working with FastAPI UploadFile objects
and converting between various data formats (PIL Image, numpy arrays, etc.).
"""

import io
import os
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import UploadFile
from PIL import Image
from PIL import ImageDraw


def ensure_image(path: str) -> Image.Image:
    """
    Load an image from path or create a placeholder if it doesn't exist.

    Creates an in-memory placeholder image (no disk writes) when the path
    doesn't exist, useful for testing and development.

    Args:
        path: Path to image file

    Returns:
        PIL Image in RGB mode

    Example:
        >>> img = ensure_image("photo.jpg")
        >>> # or placeholder if file doesn't exist
        >>> img = ensure_image("missing.jpg")
    """
    if os.path.exists(path):
        return Image.open(path).convert("RGB")
    else:
        # Create placeholder image in memory (no disk write)
        img = Image.new("RGB", (768, 512), "#E8F2FF")
        d = ImageDraw.Draw(img)
        d.rectangle((20, 400, 300, 500), fill="#F4F4F4", outline="#CCCCCC")
        d.text(
            (30, 410), f"placeholder {os.path.basename(path)}", fill="#333333"
        )
        return img


def get_upload_file_image(
    upload_file: Optional[UploadFile],
) -> Optional[Image.Image]:
    """
    Convert FastAPI UploadFile to PIL Image.

    Reads the file contents and resets the file pointer for potential re-reading.

    Args:
        upload_file: Optional FastAPI UploadFile object

    Returns:
        PIL Image in RGB mode, or None if upload_file is None

    Example:
        >>> from fastapi import UploadFile
        >>> img = get_upload_file_image(upload_file)
        >>> if img:
        ...     processed = model(img)
    """
    if upload_file is None:
        return None
    contents = upload_file.file.read()
    upload_file.file.seek(0)  # Reset for potential re-reading
    return Image.open(io.BytesIO(contents)).convert("RGB")


def get_upload_file_path(
    upload_file: Optional[UploadFile], temp_path: str
) -> Optional[str]:
    """
    Save UploadFile to temporary path and return the path.

    Creates parent directories if needed and resets file pointer after writing.

    Args:
        upload_file: Optional FastAPI UploadFile object
        temp_path: Destination path for temporary file

    Returns:
        Path where file was saved, or None if upload_file is None

    Example:
        >>> temp_path = f"/tmp/video_{os.getpid()}.mp4"
        >>> saved_path = get_upload_file_path(upload_file, temp_path)
        >>> if saved_path:
        ...     process_video(saved_path)
    """
    if upload_file is None:
        return None
    os.makedirs(os.path.dirname(temp_path) or ".", exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(upload_file.file.read())
    upload_file.file.seek(0)  # Reset for potential re-reading
    return temp_path


def image_to_bytes(img: Image.Image, img_format: str = "PNG") -> bytes:
    """
    Convert PIL Image to bytes.

    Args:
        img: PIL Image object
        img_format: Output format (e.g., 'PNG', 'JPEG')

    Returns:
        Image data as bytes

    Example:
        >>> img = Image.new("RGB", (100, 100), "blue")
        >>> data = image_to_bytes(img, "PNG")
        >>> # Save or stream data
    """
    buf = io.BytesIO()
    img.save(buf, format=img_format)
    return buf.getvalue()


def audio_to_bytes(audio: np.ndarray, sr: int) -> bytes:
    """
    Convert audio array to WAV bytes.

    Handles both mono and stereo audio, transposing if needed.

    Args:
        audio: Numpy array of audio samples
        sr: Sample rate in Hz

    Returns:
        WAV file data as bytes

    Example:
        >>> import numpy as np
        >>> audio = np.random.randn(16000)  # 1 second at 16kHz
        >>> wav_data = audio_to_bytes(audio, 16000)
    """
    arr = np.asarray(audio).squeeze()
    # If (channels, samples) flip to (samples, channels)
    if arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
        arr = arr.T
    buf = io.BytesIO()
    sf.write(buf, arr, sr, format="WAV")
    return buf.getvalue()


__all__ = [
    "ensure_image",
    "get_upload_file_image",
    "get_upload_file_path",
    "image_to_bytes",
    "audio_to_bytes",
]
