from transformers import pipeline
from app.helpers import device_arg, ensure_image, safe_print_output
from app.utilities import is_gated_repo_error, is_missing_model_error, soft_skip

def run_object_detection(spec, dev: str):
    img = ensure_image(spec["payload"]["image_path"])
    try:
        pl = pipeline("object-detection", model=spec["model_id"], device=device_arg(dev))
        out = pl(img)
        safe_print_output(out)
    except Exception as e:
        if is_gated_repo_error(e): soft_skip("gated model (no access/auth)"); return
        if is_missing_model_error(e): soft_skip("model not found on Hugging Face"); return
        safe_print_output({"error": "object-detection failed", "reason": repr(e)})
