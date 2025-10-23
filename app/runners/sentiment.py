"""Sentiment analysis runner (pure function with explicit error handling).

This runner demonstrates functional patterns:
- Pure function (given same inputs, returns same outputs)
- No hidden state or mutations
- Explicit error handling (returns error dict rather than raising)
- Can be composed with higher-order functions

Example using HOFs (alternative approach):
    from app.infrastructure.runner_utils import with_standard_error_handling
    from app.infrastructure.runner_utils import validate_spec_fields
    
    @with_standard_error_handling
    @validate_spec_fields(["prompt"])
    def run_sentiment_functional(spec: RunnerSpec, dev: str) -> Dict[str, Any]:
        # Core logic only, error handling provided by HOFs
        prompt = spec["payload"]["prompt"]
        pl = pipeline("text-classification", model=spec["model_id"], device=dev)
        return safe_json(pl(prompt))
"""
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
    Run sentiment analysis (text classification) inference (pure function).
    
    This is a pure function: given the same spec and device, it will
    always return the same result (modulo external API calls).
    
    Args:
        spec: Immutable spec containing model_id, payload, extra_args
        dev: Device string (e.g., "cpu", "cuda:0")
    
    Returns:
        Dict with result or error/skipped information
    
    Side effects:
        - Calls external transformers API (impure)
        - May download model from HuggingFace (I/O)
    """
    extra_args: Dict[str, Any] = spec.get("extra_args", {}) or {}

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
            **extra_args,
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
