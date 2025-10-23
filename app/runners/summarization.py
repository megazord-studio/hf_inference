from typing import Any
from typing import Dict

from transformers import pipeline

from app.infrastructure.device import device_arg
from app.infrastructure.response import safe_json
from app.types import RunnerSpec
from app.infrastructure.errors import is_gated_repo_error
from app.infrastructure.errors import is_missing_model_error


def run_summarization(spec: RunnerSpec, dev: str) -> Dict[str, Any]:
    """
    Run summarization inference.
    Returns the result as a dictionary instead of printing.
    """
    extra_args: Dict[str, Any] = spec.get("extra_args", {}) or {}

    try:
        pl = pipeline(
            "summarization",
            model=spec["model_id"],
            device=device_arg(dev),
            **extra_args,
        )
        out = pl(spec["payload"]["prompt"], max_new_tokens=64)
        return safe_json(out)
    except Exception as e:
        if is_gated_repo_error(e):
            return {"skipped": True, "reason": "gated model (no access/auth)"}
        if is_missing_model_error(e):
            return {
                "skipped": True,
                "reason": "model not found on Hugging Face",
            }
        return {"error": "summarization failed", "reason": repr(e)}
