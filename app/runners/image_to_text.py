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


def run_image_to_text(spec: RunnerSpec, dev: str) -> Dict[str, Any]:
    """
    Run image-to-text inference.
    Accepts either image_path or UploadFile from spec["files"]["image"].
    Returns the result as a dictionary instead of printing.
    """
    # Handle UploadFile or fallback to path
    img = get_upload_file_image(spec.get("files", {}).get("image"))
    if img is None:
        img = ensure_image(spec["payload"].get("image_path", "image.jpg"))

    try:
        pl = pipeline(
            "image-to-text",
            model=spec["model_id"],
            device=device_arg(dev),
            trust_remote_code=True,
        )
        out = pl(img)
        if isinstance(out, list) and out and "generated_text" in out[0]:
            return {"text": out[0]["generated_text"]}
        else:
            return safe_json(out)
    except Exception as e:
        if is_gated_repo_error(e):
            return {
                "skipped": True,
                "reason": "gated model (no access/auth)",
                "hint": "Try nlpconnect/vit-gpt2-image-captioning.",
            }
        if is_missing_model_error(e):
            return {
                "skipped": True,
                "reason": "model not found on Hugging Face",
                "hint": "Try nlpconnect/vit-gpt2-image-captioning.",
            }
        cap = _final_caption_fallback(img, dev)
        return (
            cap
            if "text" not in cap
            else {"text": cap["text"], "note": "fallback caption used"}
        )
