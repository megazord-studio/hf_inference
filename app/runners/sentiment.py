from transformers import pipeline
from app.helpers import device_arg, safe_json
from app.utilities import is_gated_repo_error, is_missing_model_error

def run_sentiment(spec, dev: str):
    """
    Run sentiment analysis inference.
    Returns the result as a dictionary instead of printing.
    """
    try:
        pl = pipeline("sentiment-analysis", model=spec["model_id"], device=device_arg(dev))
        out = pl(spec["payload"]["prompt"])
        return safe_json(out)
    except Exception as e:
        if is_gated_repo_error(e):
            return {"skipped": True, "reason": "gated model (no access/auth)"}
        if is_missing_model_error(e):
            return {"skipped": True, "reason": "model not found on Hugging Face"}
        return {"error": "sentiment-analysis failed", "reason": repr(e)}
