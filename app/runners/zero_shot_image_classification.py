from transformers import pipeline
from app.helpers import device_arg, ensure_image, safe_print_output
from app.utilities import is_gated_repo_error, is_missing_model_error, soft_skip

def run_zero_shot_image_classification(spec, dev: str):
    img = ensure_image(spec["payload"]["image_path"])
    try:
        pl = pipeline("zero-shot-image-classification", model=spec["model_id"], device=device_arg(dev))
        out = pl(img, candidate_labels=spec["payload"]["candidate_labels"])
        safe_print_output(out)
    except Exception as e:
        if is_gated_repo_error(e): soft_skip("gated model (no access/auth)"); return
        if is_missing_model_error(e):
            soft_skip("model not found on Hugging Face", "Use openai/clip-vit-base-patch32 or laion/CLIP-ViT-H-14."); return
        safe_print_output({"error": "zero-shot-image-classification failed", "reason": repr(e)})
