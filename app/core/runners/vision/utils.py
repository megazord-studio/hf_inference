from __future__ import annotations

import base64
import io
import logging

from PIL import Image

log = logging.getLogger("app.runners.vision")


def decode_base64_image(b64: str) -> Image.Image:
    """Decode base64 image to PIL Image.

    Accepts both full data URIs (e.g., "data:image/png;base64,....") and
    bare base64 payloads ("iVBOR...").
    """
    payload = b64.split(",", 1)[1] if "," in b64 else b64
    img_bytes = base64.b64decode(payload)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")
