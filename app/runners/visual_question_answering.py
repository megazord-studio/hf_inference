from transformers import pipeline
from app.helpers import device_arg, ensure_image, safe_print_output
from app.utilities import is_gated_repo_error, is_missing_model_error, soft_hint_error, soft_skip, _final_caption_fallback

def run_vqa(spec, dev: str):
    img = ensure_image(spec["payload"]["image_path"])
    try:
        pl = pipeline("visual-question-answering", model=spec["model_id"], device=device_arg(dev), trust_remote_code=True)
        out = pl(image=img, question=spec["payload"]["question"])
        safe_print_output(out)
    except Exception as e:
        if is_gated_repo_error(e):
            soft_skip("gated model (no access/auth)", "Try dandelin/vilt-b32-finetuned-vqa."); return
        if is_missing_model_error(e):
            soft_skip("model not found on Hugging Face", "Try dandelin/vilt-b32-finetuned-vqa."); return
        cap = _final_caption_fallback(img, dev)
        if "text" in cap:
            safe_print_output({"text": cap["text"], "note": "fallback caption used"})
        else:
            soft_hint_error("visual-question-answering failed", repr(e))
