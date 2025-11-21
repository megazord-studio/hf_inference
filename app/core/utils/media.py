"""Media base64 utilities (images) for vision generation runners.

DRY: centrally encode/decode to keep runners focused on inference only.
"""
from __future__ import annotations
from typing import Tuple
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

__all__ = ["decode_image_base64", "encode_image_base64", "image_size", "IMG_PREFIX"]

