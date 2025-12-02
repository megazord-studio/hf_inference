from __future__ import annotations

import base64
import io
import logging

from PIL import Image

log = logging.getLogger("app.runners.vision")


def decode_base64_image(b64: str) -> Image.Image:
    """Decode base64 image to PIL Image."""
    header, data = b64.split(",", 1)
    img_bytes = base64.b64decode(data)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")
