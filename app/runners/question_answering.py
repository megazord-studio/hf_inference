from typing import Any
from typing import Dict

from transformers import pipeline

from app.helpers import device_arg
from app.helpers import safe_json
from app.types import RunnerSpec
from app.utilities import is_gated_repo_error
from app.utilities import is_missing_model_error


def run_qa(spec: RunnerSpec, dev: str) -> Dict[str, Any]:
    """
    Run question answering inference.
    Returns the result as a dictionary instead of printing.
    """
    try:
        pl = pipeline(
            "question-answering",
            model=spec["model_id"],
            device=device_arg(dev),
        )
        p = spec["payload"]
        out = pl(question=p["qa_question"], context=p["qa_context"])
        return safe_json(out)
    except Exception as e:
        if is_gated_repo_error(e):
            return {"skipped": True, "reason": "gated model (no access/auth)"}
        if is_missing_model_error(e):
            return {
                "skipped": True,
                "reason": "model not found on Hugging Face",
            }
        return {"error": "question-answering failed", "reason": repr(e)}
