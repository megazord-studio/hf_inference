from transformers import pipeline
from app.helpers import device_arg, ensure_image, safe_print_output
from app.utilities import _vlm_llava, _vlm_florence2, _vlm_minicpm, _final_caption_fallback

def run_vlm_image_text_to_text(spec, dev: str):
    payload = spec["payload"]
    img = ensure_image(payload["image_path"])
    prompt = payload.get("prompt", "Describe the image briefly.")
    mid = spec["model_id"].lower()

    if "llava" in mid:
        safe_print_output(_vlm_llava(spec, img, prompt, dev)); return
    if "florence-2" in mid or "florence" in mid:
        safe_print_output(_vlm_florence2(spec, img, prompt, dev)); return
    if "minicpm" in mid or "cpm" in mid:
        safe_print_output(_vlm_minicpm(spec, img, prompt, dev)); return

    try:
        pl = pipeline("image-text-to-text", model=spec["model_id"], trust_remote_code=True, device=device_arg(dev))
        out = pl({"image": img, "prompt": prompt})
        if isinstance(out, dict) and "text" in out:
            safe_print_output({"text": out["text"]}); return
        if isinstance(out, list) and out and "generated_text" in out[0]:
            safe_print_output({"text": out[0]["generated_text"]}); return
        if isinstance(out, str):
            safe_print_output({"text": out}); return
    except Exception:
        pass
    try:
        pl2 = pipeline("image-to-text", model=spec["model_id"], trust_remote_code=True, device=device_arg(dev))
        out2 = pl2(img)
        if isinstance(out2, list) and out2 and "generated_text" in out2[0]:
            safe_print_output({"text": out2[0]["generated_text"]}); return
        if isinstance(out2, str):
            safe_print_output({"text": out2}); return
    except Exception:
        pass
    cap = _final_caption_fallback(img, dev)
    if "text" in cap:
        safe_print_output({"text": cap["text"], "note": "fallback caption used"})
    else:
        safe_print_output(cap)
