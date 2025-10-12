from transformers import pipeline
from app.helpers import device_arg, ensure_image, safe_print_output
from app.utilities import is_gated_repo_error, is_missing_model_error, soft_skip, _final_caption_fallback

def run_image_to_text(spec, dev: str):
    img = ensure_image(spec["payload"]["image_path"])
    try:
        pl = pipeline("image-to-text", model=spec["model_id"], device=device_arg(dev), trust_remote_code=True)
        out = pl(img)
        if isinstance(out, list) and out and "generated_text" in out[0]:
            safe_print_output({"text": out[0]["generated_text"]})
        else:
            safe_print_output(out)
    except Exception as e:
        if is_gated_repo_error(e): soft_skip("gated model (no access/auth)", "Try nlpconnect/vit-gpt2-image-captioning."); return
        if is_missing_model_error(e): soft_skip("model not found on Hugging Face", "Try nlpconnect/vit-gpt2-image-captioning."); return
        cap = _final_caption_fallback(img, dev)
        safe_print_output(cap if "text" not in cap else {"text": cap["text"], "note": "fallback caption used"})
