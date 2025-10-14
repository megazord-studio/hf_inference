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


def run_doc_qa(spec: RunnerSpec, dev: str) -> Dict[str, Any]:
    """
    Run document question answering inference.
    Accepts either image_path or UploadFile from spec["files"]["image"].
    Returns the result as a dictionary instead of printing.
    """
    # Handle UploadFile or fallback to path
    img = get_upload_file_image(spec.get("files", {}).get("image"))
    if img is None:
        img = ensure_image(spec["payload"].get("image_path", "image.jpg"))

    mid = spec["model_id"].lower()
    if "layoutlmv3" in mid:
        return {
            "skipped": True,
            "reason": "model incompatible with document-question-answering pipeline",
            "hint": "Use a doc-VQA model like impira/layoutlm-document-qa or naver-clova-ix/donut-base-finetuned-docvqa.",
        }
    try:
        pl = pipeline(
            "document-question-answering",
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
                "hint": "Pick a published doc-VQA model, e.g., impira/layoutlm-document-qa or donut docvqa.",
            }
        if is_missing_model_error(e):
            return {
                "skipped": True,
                "reason": "model not found on Hugging Face",
                "hint": "Pick a published doc-VQA model, e.g., impira/layoutlm-document-qa or donut docvqa.",
            }
        return {
            "error": "document-question-answering failed",
            "reason": repr(e),
            "hint": "Ensure the model supports doc-qa or switch to a compatible doc-VQA model.",
        }
