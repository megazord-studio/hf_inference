from transformers import pipeline
from app.helpers import device_arg, safe_print_output
from app.utilities import is_gated_repo_error, is_missing_model_error, soft_skip

def run_ner(spec, dev: str):
    try:
        pl = pipeline("token-classification", model=spec["model_id"], aggregation_strategy="simple", device=device_arg(dev))
        out = pl(spec["payload"]["prompt"])
        for o in out:
            o["score"] = float(o.get("score", 0.0))
        safe_print_output(out)
    except Exception as e:
        if is_gated_repo_error(e): soft_skip("gated model (no access/auth)"); return
        if is_missing_model_error(e): soft_skip("model not found on Hugging Face"); return
        safe_print_output({"error": "token-classification failed", "reason": repr(e)})
