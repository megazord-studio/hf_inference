from typing import Any
from typing import Dict

from transformers import pipeline
from transformers.pipelines import TextClassificationPipeline

from app.helpers import device_arg
from app.helpers import safe_json
from app.types import RunnerSpec
from app.utilities import is_gated_repo_error
from app.utilities import is_missing_model_error


def run_sentiment(spec: RunnerSpec, dev: str) -> Dict[str, Any]:
    """
    Run sentiment analysis (text classification) inference.
    Returns the result as a dictionary instead of printing.
    """
    try:
        prompt = str(spec.get("payload", {}).get("prompt", "")).strip()
        if not prompt:
            return {
                "error": "sentiment-analysis failed",
                "reason": "empty prompt",
            }

        pl: TextClassificationPipeline = pipeline(
            task="text-classification",
            model=spec["model_id"],
            device=device_arg(dev),
        )

        out = pl(prompt)
        return safe_json(out)
    except Exception as e:
        if is_gated_repo_error(e):
            return {"skipped": True, "reason": "gated model (no access/auth)"}
        if is_missing_model_error(e):
            return {
                "skipped": True,
                "reason": "model not found on Hugging Face",
            }
        return {"error": "sentiment-analysis failed", "reason": repr(e)}
