from transformers import pipeline
from app.helpers import device_arg, safe_print_output
from app.utilities import is_gated_repo_error, is_missing_model_error, soft_skip

def run_summarization(spec, dev: str):
    try:
        pl = pipeline("summarization", model=spec["model_id"], device=device_arg(dev))
        out = pl(spec["payload"]["prompt"], max_new_tokens=64)
        safe_print_output(out)
    except Exception as e:
        if is_gated_repo_error(e): soft_skip("gated model (no access/auth)"); return
        if is_missing_model_error(e): soft_skip("model not found on Hugging Face"); return
        safe_print_output({"error": "summarization failed", "reason": repr(e)})
