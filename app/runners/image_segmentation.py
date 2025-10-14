import base64
from typing import Any
from typing import Dict

import numpy as np
import torch
from PIL import Image
from transformers import pipeline

from app.helpers import device_arg
from app.helpers import ensure_image
from app.helpers import get_upload_file_image
from app.helpers import image_to_bytes
from app.helpers import safe_json
from app.types import RunnerSpec
from app.utilities import is_gated_repo_error
from app.utilities import is_missing_model_error


def _convert_masks_to_base64(seg_list: Any) -> Any:
    """Convert mask images to base64 encoded strings for JSON response."""
    result: Any = []
    for seg in seg_list:
        mask = seg.get("mask", None)
        entry: Dict[str, Any] = {}
        for k, v in seg.items():
            if k == "mask":
                continue
            if k == "score":
                try:
                    entry[k] = float(v)
                except Exception:
                    entry[k] = v
            else:
                entry[k] = v

        if isinstance(mask, (Image.Image, np.ndarray, torch.Tensor)):
            if isinstance(mask, torch.Tensor):
                mask = mask.detach().cpu().numpy()
            if isinstance(mask, np.ndarray):
                from PIL import Image as PILImage

                mask_img = (
                    PILImage.fromarray((mask * 255).astype(np.uint8))
                    if mask.ndim == 2
                    else PILImage.fromarray(mask)
                )
            else:
                mask_img = mask

            # Convert mask to base64
            mask_bytes = image_to_bytes(mask_img, img_format="PNG")
            mask_b64 = base64.b64encode(mask_bytes).decode("utf-8")
            entry["mask_base64"] = mask_b64

        result.append(entry)
    return result


def run_image_segmentation(spec: RunnerSpec, dev: str) -> Dict[str, Any]:
    """
    Run image segmentation inference.
    Accepts either image_path or UploadFile from spec["files"]["image"].
    Returns masks as base64 encoded strings in JSON.
    """
    # Handle UploadFile or fallback to path
    img = get_upload_file_image(spec.get("files", {}).get("image"))
    if img is None:
        img = ensure_image(spec["payload"].get("image_path", "image.jpg"))

    try:
        pl = pipeline(
            "image-segmentation",
            model=spec["model_id"],
            device=device_arg(dev),
        )
        out = pl(img)
        masks_with_b64 = _convert_masks_to_base64(out)
        return safe_json({"masks": masks_with_b64})
    except Exception as e:
        if is_gated_repo_error(e):
            return {"skipped": True, "reason": "gated model (no access/auth)"}
        if is_missing_model_error(e):
            return {
                "skipped": True,
                "reason": "model not found on Hugging Face",
            }
        return {"error": "image-segmentation failed", "reason": repr(e)}
