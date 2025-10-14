import io
import json
import os
from typing import Any
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from fastapi import UploadFile
from PIL import Image
from PIL import ImageDraw

# ---------- device & printing ----------


def device_str() -> str:
    """Return the device string for PyTorch ('cuda:0' or 'cpu')."""
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def device_arg(dev: str) -> str:
    """Return device argument for transformers (accepts device str, torch.device, or int)."""
    # transformers >=4.41 accepts device str / torch.device / int
    return dev


def safe_json(obj: Any) -> Any:
    """
    Convert Python objects to JSON-serializable format.

    Handles numpy arrays, torch tensors, and other non-serializable types.
    """
    if isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_json(v) for v in obj]
    if isinstance(obj, (str, int, bool, type(None), float)):
        return obj
    if isinstance(obj, (np.generic,)):
        return float(obj.item())
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    try:
        import torch

        if isinstance(obj, torch.Tensor):
            if obj.numel() == 1:
                return float(obj.item())
            return obj.detach().cpu().numpy().tolist()
    except Exception:
        pass
    return str(obj)


def safe_print_output(obj: Any) -> None:
    """Print object as formatted JSON, converting non-serializable types."""
    clean = safe_json(obj)
    print(f"Output type: {type(clean)}")
    print(json.dumps(clean, indent=2, ensure_ascii=False))


# ---------- I/O helpers ----------


def ensure_image(path: str) -> Image.Image:
    """Load an image from path or URL, converting to RGB."""
    """
    Get an image from path, or create a placeholder in memory if it doesn't exist.
    Does not save anything to disk.
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


def to_dataframe(table_like: List[List[str]]) -> pd.DataFrame:
    rows = [[str(x) for x in r] for r in table_like]
    if rows and all(isinstance(c, str) for c in rows[0]):
        header = rows[0]
        data = rows[1:] if len(rows) > 1 else []
        return pd.DataFrame(data, columns=header)
    return pd.DataFrame(rows)


# ---------- File handling for FastAPI ----------


def get_upload_file_image(
    upload_file: Optional[UploadFile],
) -> Optional[Image.Image]:
    """Convert UploadFile to PIL Image."""
    if upload_file is None:
        return None
    contents = upload_file.file.read()
    upload_file.file.seek(0)  # Reset for potential re-reading
    return Image.open(io.BytesIO(contents)).convert("RGB")


def get_upload_file_path(
    upload_file: Optional[UploadFile], temp_path: str
) -> Optional[str]:
    """Save UploadFile to temporary path and return path."""
    if upload_file is None:
        return None
    os.makedirs(os.path.dirname(temp_path) or ".", exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(upload_file.file.read())
    upload_file.file.seek(0)  # Reset for potential re-reading
    return temp_path


def image_to_bytes(img: Image.Image, img_format: str = "PNG") -> bytes:
    """Convert PIL Image to bytes."""
    buf = io.BytesIO()
    img.save(buf, format=img_format)
    return buf.getvalue()


def audio_to_bytes(audio: np.ndarray, sr: int) -> bytes:
    """Convert audio array to WAV bytes."""
    arr = np.asarray(audio).squeeze()
    # If (channels, samples) flip to (samples, channels)
    if arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
        arr = arr.T
    buf = io.BytesIO()
    sf.write(buf, arr, sr, format="WAV")
    return buf.getvalue()
