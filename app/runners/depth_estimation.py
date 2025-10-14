from typing import Any
from typing import Dict

import torch
from PIL import Image as PILImage
from transformers import pipeline

from app.helpers import device_arg
from app.helpers import ensure_image
from app.helpers import get_upload_file_image
from app.helpers import image_to_bytes
from app.types import RunnerSpec
from app.utilities import is_gated_repo_error
from app.utilities import is_missing_model_error


def run_depth_estimation(spec: RunnerSpec, dev: str) -> Dict[str, Any]:
    """
    Run depth estimation inference.
    Accepts either image_path or UploadFile from spec["files"]["image"].
    Returns depth map as bytes or JSON.
    """
    # Handle UploadFile or fallback to path
    img = get_upload_file_image(spec.get("files", {}).get("image"))
    if img is None:
        img = ensure_image(spec["payload"].get("image_path", "image.jpg"))

    try:
        pl = pipeline(
            "depth-estimation", model=spec["model_id"], device=device_arg(dev)
        )
        out = pl(img)
        depth = (
            out.get("predicted_depth", None) if isinstance(out, dict) else None
        )
        if isinstance(depth, torch.Tensor):
            d = depth.detach().cpu().numpy()
            d = d - d.min()
            d = d / (d.max() if d.max() > 0 else 1)
            d_img = PILImage.fromarray((d * 255).astype("uint8"))

            # Convert to bytes
            img_bytes = image_to_bytes(d_img, img_format="PNG")
            return {
                "file_data": img_bytes,
                "file_name": f"depth_{spec['model_id'].replace('/', '_')}.png",
                "content_type": "image/png",
            }
        else:
            return out
    except Exception as e:
        if is_gated_repo_error(e):
            return {"skipped": True, "reason": "gated model (no access/auth)"}
        if is_missing_model_error(e):
            return {
                "skipped": True,
                "reason": "model not found on Hugging Face",
            }
        return {"error": "depth-estimation failed", "reason": repr(e)}
