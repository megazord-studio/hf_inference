from transformers import pipeline

from app.helpers import device_arg
from app.helpers import safe_json
from app.utilities import is_gated_repo_error
from app.utilities import is_missing_model_error


def run_ner(spec, dev: str):
    """
    Run token classification (NER) inference.
    Returns the result as a dictionary instead of printing.
    """
    try:
        pl = pipeline(
            "token-classification",
            model=spec["model_id"],
            aggregation_strategy="simple",
            device=device_arg(dev),
        )
        out = pl(spec["payload"]["prompt"])
        for o in out:
            o["score"] = float(o.get("score", 0.0))
        return safe_json(out)
    except Exception as e:
        if is_gated_repo_error(e):
            return {"skipped": True, "reason": "gated model (no access/auth)"}
        if is_missing_model_error(e):
            return {
                "skipped": True,
                "reason": "model not found on Hugging Face",
            }
        return {"error": "token-classification failed", "reason": repr(e)}
