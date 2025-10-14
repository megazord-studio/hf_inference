from typing import Any
from typing import Dict

import numpy as np
import torch
from transformers import AutoModel
from transformers import AutoProcessor
from transformers import pipeline

from app.helpers import device_arg
from app.helpers import device_str
from app.helpers import ensure_image
from app.helpers import get_upload_file_image
from app.types import RunnerSpec
from app.utilities import is_gated_repo_error
from app.utilities import is_missing_model_error


def run_image_feature_extraction(spec: RunnerSpec, dev: str) -> Dict[str, Any]:
    """
    Run image feature extraction inference.
    Accepts either image_path or UploadFile from spec["files"]["image"].
    Returns the result as a dictionary instead of printing.
    """
    # Handle UploadFile or fallback to path
    img = get_upload_file_image(spec.get("files", {}).get("image"))
    if img is None:
        img = ensure_image(spec["payload"].get("image_path", "image.jpg"))

    try:
        if "clip" in spec["model_id"].lower():
            proc = AutoProcessor.from_pretrained(spec["model_id"])  # nosec
            model = AutoModel.from_pretrained(spec["model_id"]).to(
                device_str()
            )
            inputs = proc(images=img, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                feats = model.get_image_features(**inputs)
            return {"embedding_shape": tuple(feats.shape)}
        pl = pipeline(
            "image-feature-extraction",
            model=spec["model_id"],
            device=device_arg(dev),
        )
        feats = pl(img)
        return {"embedding_shape": np.array(feats).shape}
    except Exception as e:
        if is_gated_repo_error(e):
            return {"skipped": True, "reason": "gated model (no access/auth)"}
        if is_missing_model_error(e):
            return {
                "skipped": True,
                "reason": "model not found on Hugging Face",
            }
        return {"error": "image-feature-extraction failed", "reason": repr(e)}
