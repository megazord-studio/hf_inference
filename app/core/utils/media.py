"""Media base64 utilities (images, lightweight GIF video) for runners.

DRY: centrally encode/decode to keep runners focused on inference only.
"""
from __future__ import annotations
from typing import Tuple, List
from PIL import Image
import base64, io

IMG_PREFIX = "data:image/png;base64,"

def decode_image_base64(data: str) -> Image.Image:
    if ',' in data:
        _, b64 = data.split(',', 1)
    else:
        b64 = data
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert('RGB')

def encode_image_base64(img: Image.Image, format: str = 'PNG') -> str:
    buf = io.BytesIO()
    img.save(buf, format=format)
    return IMG_PREFIX + base64.b64encode(buf.getvalue()).decode()

def image_size(img: Image.Image) -> Tuple[int, int]:
    return img.width, img.height

# --- Lightweight video (GIF) helpers for legacy video runners -----------------

VIDEO_GIF_PREFIX = "data:image/gif;base64,"


def encode_gif_base64(frames: List[Image.Image], duration_ms: int = 200, loop: int = 0) -> str:
    """Encode a small list of PIL Image frames as an animated GIF data URI.

    Kept for potential backwards compatibility; current Phase F runners use
    real video models and MP4 encoding instead.
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
    if "," in data:
        _, b64 = data.split(",", 1)
    else:
        b64 = data
    raw = base64.b64decode(b64)
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
    "VIDEO_GIF_PREFIX",
    "encode_gif_base64",
    "decode_gif_base64",
]
