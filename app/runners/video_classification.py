from transformers import pipeline
from app.helpers import device_arg, safe_print_output
from app.utilities import is_gated_repo_error, is_missing_model_error, soft_skip

def run_video_classification(spec, dev: str):
    try:
        pl = pipeline("video-classification", model=spec["model_id"], device=device_arg(dev))
        out = pl(spec["payload"]["video_path"])
        if isinstance(out, list):
            for o in out:
                if "score" in o:
                    o["score"] = float(o["score"])
        safe_print_output(out)
    except Exception as e:
        if "requires the PyAv library" in repr(e):
            soft_skip("missing dependency: PyAV", "Install with: pip install av"); return
        if is_gated_repo_error(e): soft_skip("gated model (no access/auth)"); return
        if is_missing_model_error(e): soft_skip("model not found on Hugging Face"); return
        safe_print_output({"error": "video-classification failed", "reason": repr(e),
                           "hint": "This pipeline may need decord/pyav (pip install av) and GPU-appropriate deps."})
