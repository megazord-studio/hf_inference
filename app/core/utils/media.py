"""Media base64 utilities (images, audio, video, 3D) for runners.

DRY: centrally encode/decode to keep runners focused on inference only.
"""
from __future__ import annotations
from typing import Tuple, List
from PIL import Image
import base64, io

IMG_PREFIX = "data:image/png;base64,"
AUDIO_PREFIX = "data:audio/wav;base64,"
VIDEO_PREFIX = "data:video/mp4;base64,"
THREED_PREFIX = "data:model/gltf-binary;base64,"


def _strip_data_prefix(data: str) -> str:
    """Return the base64 payload, accepting full data URIs or bare base64.

    This keeps runners tolerant to both forms while tests use explicit
    data: prefixes.
    """
    if "," in data:
        _, b64 = data.split(",", 1)
        return b64
    return data


def decode_image_base64(data: str) -> Image.Image:
    raw = base64.b64decode(_strip_data_prefix(data))
    return Image.open(io.BytesIO(raw)).convert('RGB')


def encode_image_base64(img: Image.Image, format: str = 'PNG') -> str:
    buf = io.BytesIO()
    img.save(buf, format=format)
    return IMG_PREFIX + base64.b64encode(buf.getvalue()).decode()


def image_size(img: Image.Image) -> Tuple[int, int]:
    return img.width, img.height


# --- Audio helpers -----------------------------------------------------------


def encode_audio_base64(wav_bytes: bytes, mime: str = "audio/wav") -> str:
    prefix = f"data:{mime};base64,"
    return prefix + base64.b64encode(wav_bytes).decode()


def decode_audio_base64(data: str) -> bytes:
    """Decode audio data URI or bare base64 into raw bytes.

    Runners are responsible for converting these bytes into tensors using
    their chosen audio library (torchaudio, soundfile, etc.).
    """
    return base64.b64decode(_strip_data_prefix(data))


# --- Video helpers (MP4 bytes) ----------------------------------------------


def encode_video_base64(video_bytes: bytes, mime: str = "video/mp4") -> str:
    prefix = f"data:{mime};base64,"
    return prefix + base64.b64encode(video_bytes).decode()


def decode_video_base64(data: str) -> bytes:
    return base64.b64decode(_strip_data_prefix(data))


# --- 3D (GLB) helpers --------------------------------------------------------


def encode_3d_base64(glb_bytes: bytes, mime: str = "model/gltf-binary") -> str:
    prefix = f"data:{mime};base64,"
    return prefix + base64.b64encode(glb_bytes).decode()


def decode_3d_base64(data: str) -> bytes:
    return base64.b64decode(_strip_data_prefix(data))


# --- Lightweight GIF helpers (legacy) ---------------------------------------

VIDEO_GIF_PREFIX = "data:image/gif;base64,"


def encode_gif_base64(frames: List[Image.Image], duration_ms: int = 200, loop: int = 0) -> str:
    """Encode a small list of PIL Image frames as an animated GIF data URI.

    Kept for potential backwards compatibility; main video path uses MP4.
    """
    if not frames:
        raise ValueError("encode_gif_base64: no frames provided")
    buf = io.BytesIO()
    first, *rest = frames
    first.save(
        buf,
        format="GIF",
        save_all=True,
        append_images=rest,
        duration=duration_ms,
        loop=loop,
    )
    return VIDEO_GIF_PREFIX + base64.b64encode(buf.getvalue()).decode()


def decode_gif_base64(data: str) -> List[Image.Image]:
    """Decode a GIF data URI or raw base64 string into a list of frames."""
    raw = base64.b64decode(_strip_data_prefix(data))
    im = Image.open(io.BytesIO(raw))
    frames: List[Image.Image] = []
    try:
        while True:
            frames.append(im.copy().convert("RGB"))
            im.seek(im.tell() + 1)
    except EOFError:
        pass
    if not frames:
        frames.append(im.copy().convert("RGB"))
    return frames

__all__ = [
    "decode_image_base64",
    "encode_image_base64",
    "image_size",
    "IMG_PREFIX",
    "AUDIO_PREFIX",
    "VIDEO_PREFIX",
    "THREED_PREFIX",
    "encode_audio_base64",
    "decode_audio_base64",
    "encode_video_base64",
    "decode_video_base64",
    "encode_3d_base64",
    "decode_3d_base64",
    "VIDEO_GIF_PREFIX",
    "encode_gif_base64",
    "decode_gif_base64",
]
