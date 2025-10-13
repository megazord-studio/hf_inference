from transformers import pipeline
from app.helpers import device_arg, ensure_image, get_upload_file_image
from app.utilities import _vlm_llava, _vlm_florence2, _vlm_minicpm, _final_caption_fallback

def run_vlm_image_text_to_text(spec, dev: str):
    """
    Run vision-language model image-text-to-text inference.
    Accepts either image_path or UploadFile from spec["files"]["image"].
    Returns the result as a dictionary instead of printing.
    """
    payload = spec["payload"]
    
    # Handle UploadFile or fallback to path
    img = get_upload_file_image(spec.get("files", {}).get("image"))
    if img is None:
        img = ensure_image(payload.get("image_path", "image.jpg"))
    
    prompt = payload.get("prompt", "Describe the image briefly.")
    mid = spec["model_id"].lower()

    if "llava" in mid:
        return _vlm_llava(spec, img, prompt, dev)
    if "florence-2" in mid or "florence" in mid:
        return _vlm_florence2(spec, img, prompt, dev)
    if "minicpm" in mid or "cpm" in mid:
        return _vlm_minicpm(spec, img, prompt, dev)

    try:
        pl = pipeline("image-text-to-text", model=spec["model_id"], trust_remote_code=True, device=device_arg(dev))
        out = pl({"image": img, "prompt": prompt})
        if isinstance(out, dict) and "text" in out:
            return {"text": out["text"]}
        if isinstance(out, list) and out and "generated_text" in out[0]:
            return {"text": out[0]["generated_text"]}
        if isinstance(out, str):
            return {"text": out}
    except Exception:
        pass
    try:
        pl2 = pipeline("image-to-text", model=spec["model_id"], trust_remote_code=True, device=device_arg(dev))
        out2 = pl2(img, max_new_tokens=20)
        if isinstance(out2, list) and out2 and "generated_text" in out2[0]:
            return {"text": out2[0]["generated_text"]}
        if isinstance(out2, str):
            return {"text": out2}
    except Exception:
        pass
    cap = _final_caption_fallback(img, dev)
    if "text" in cap:
        return {"text": cap["text"], "note": "fallback caption used"}
    else:
        return cap
