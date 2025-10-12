from transformers import pipeline
from app.helpers import device_arg, safe_print_output
from app.utilities import is_gated_repo_error, is_missing_model_error, is_no_weight_files_error, soft_skip

def run_audio_classification(spec, dev: str):
    try:
        pl = pipeline("audio-classification", model=spec["model_id"], device=device_arg(dev))
        out = pl(spec["payload"]["audio_path"])
        for o in out:
            o["score"] = float(o["score"])
        safe_print_output(out)
    except Exception as e:
        if is_gated_repo_error(e): soft_skip("gated model (no access/auth)"); return
        if is_missing_model_error(e) or is_no_weight_files_error(e):
            soft_skip("model not loadable (missing/unsupported weights)",
                      "Use superb/hubert-base-superb-er or similar."); return
        safe_print_output({"error": "audio-classification failed", "reason": repr(e)})
