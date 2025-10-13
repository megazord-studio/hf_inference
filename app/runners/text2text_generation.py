from transformers import pipeline
from app.helpers import device_arg, safe_json
from app.utilities import is_cuda_oom, is_gated_repo_error, is_missing_model_error

def run_text2text(spec, dev: str):
    """
    Run text2text generation inference.
    Returns the result as a dictionary instead of printing.
    """
    try:
        pl = pipeline("text2text-generation", model=spec["model_id"], device=device_arg(dev))
        out = pl(spec["payload"]["prompt"], max_new_tokens=64)
        return safe_json(out)
    except Exception as e:
        if is_cuda_oom(e):
            try:
                pl = pipeline("text2text-generation", model=spec["model_id"], device="cpu")
                out = pl(spec["payload"]["prompt"], max_new_tokens=64)
                return safe_json(out)
            except Exception:
                pass
        if is_gated_repo_error(e):
            return {"skipped": True, "reason": "gated model (no access/auth)"}
        if is_missing_model_error(e):
            return {"skipped": True, "reason": "model not found on Hugging Face"}
        return {"error": "text2text-generation failed", "reason": repr(e)}
