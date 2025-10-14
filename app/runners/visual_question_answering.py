from typing import Any
from typing import Dict

from transformers import pipeline

from app.helpers import device_arg
from app.helpers import ensure_image
from app.helpers import get_upload_file_image
from app.helpers import safe_json
from app.types import RunnerSpec
from app.utilities import _final_caption_fallback
from app.utilities import is_gated_repo_error
from app.utilities import is_missing_model_error


def run_vqa(spec: RunnerSpec, dev: str) -> Dict[str, Any]:
    """
    Run visual question answering inference.
    Accepts either image_path or UploadFile from spec["files"]["image"].
    Returns the result as a dictionary instead of printing.
    """
    # Handle UploadFile or fallback to path
    img = get_upload_file_image(spec.get("files", {}).get("image"))
    if img is None:
        img = ensure_image(spec["payload"].get("image_path", "image.jpg"))

    try:
        pl = pipeline(
            "visual-question-answering",
            model=spec["model_id"],
            device=device_arg(dev),
            trust_remote_code=True,
        )
        out = pl(image=img, question=spec["payload"]["question"])
        return safe_json(out)
    except Exception as e:
        if is_gated_repo_error(e):
            return {
                "skipped": True,
                "reason": "gated model (no access/auth)",
                "hint": "Try dandelin/vilt-b32-finetuned-vqa.",
            }
        if is_missing_model_error(e):
            return {
                "skipped": True,
                "reason": "model not found on Hugging Face",
                "hint": "Try dandelin/vilt-b32-finetuned-vqa.",
            }
        cap = _final_caption_fallback(img, dev)
        if "text" in cap:
            return {"text": cap["text"], "note": "fallback caption used"}
        else:
            return {
                "error": "visual-question-answering failed",
                "reason": repr(e),
            }
