from typing import Any
from typing import Dict

from transformers import pipeline

from app.helpers import device_arg
from app.helpers import ensure_image
from app.helpers import get_upload_file_image
from app.helpers import safe_json
from app.types import RunnerSpec
from app.utilities import is_gated_repo_error
from app.utilities import is_missing_model_error


def run_zero_shot_object_detection(
    spec: RunnerSpec, dev: str
) -> Dict[str, Any]:
    """
    Run zero-shot object detection inference.
    Accepts either image_path or UploadFile from spec["files"]["image"].
    Returns the result as a dictionary instead of printing.
    """
    # Handle UploadFile or fallback to path
    img = get_upload_file_image(spec.get("files", {}).get("image"))
    if img is None:
        img = ensure_image(spec["payload"].get("image_path", "image.jpg"))

    try:
        pl = pipeline(
            "zero-shot-object-detection",
            model=spec["model_id"],
            device=device_arg(dev),
        )
        out = pl(img, candidate_labels=spec["payload"]["candidate_labels"])
        return safe_json(out)
    except Exception as e:
        if is_gated_repo_error(e):
            return {"skipped": True, "reason": "gated model (no access/auth)"}
        if is_missing_model_error(e):
            return {
                "skipped": True,
                "reason": "model not found on Hugging Face",
            }
        return {
            "error": "zero-shot-object-detection failed",
            "reason": repr(e),
        }
