from typing import Any
from typing import Dict

from transformers import pipeline

from app.helpers import device_arg
from app.helpers import safe_json
from app.types import RunnerSpec
from app.utilities import is_gated_repo_error
from app.utilities import is_missing_model_error


def run_text_generation(spec: RunnerSpec, dev: str) -> Dict[str, Any]:
    """
    Run text generation inference.
    Returns the result as a dictionary instead of printing.
    """
    try:
        pl = pipeline(
            "text-generation", model=spec["model_id"], device=device_arg(dev)
        )
        out = pl(spec["payload"]["prompt"], max_new_tokens=64)
        return safe_json(out)
    except Exception as e:
        if is_gated_repo_error(e):
            return {
                "skipped": True,
                "reason": "gated model (no access/auth)",
                "hint": "Request access via HF and login.",
            }
        if is_missing_model_error(e):
            return {
                "skipped": True,
                "reason": "model not found on Hugging Face",
            }
        return {"error": "text-generation failed", "reason": repr(e)}
