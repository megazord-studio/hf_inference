import os
import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
from app.helpers import ensure_image, safe_print_output, OUT_DIR, device_str
from app.utilities import is_gated_repo_error, is_missing_model_error, soft_skip

def run_image_to_image(spec, dev: str):
    p = spec["payload"]; init_img = ensure_image(p["init_image_path"]); model_id = spec["model_id"]
    try:
        if "inpaint" in model_id.lower():
            pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device_str())
            w, h = init_img.size
            mask = Image.new("L", (w, h), 255)
            with torch.inference_mode():
                out = pipe(prompt=p["prompt"], image=init_img, mask_image=mask, guidance_scale=7.0, num_inference_steps=25)
            img = out.images[0]
        else:
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device_str())
            with torch.inference_mode():
                out = pipe(prompt=p["prompt"], image=init_img, strength=0.6, guidance_scale=7.0, num_inference_steps=25)
            img = out.images[0]
        path = os.path.join(OUT_DIR, f"sd_img2img_{model_id.replace('/','_')}.png")
        img.save(path)
        safe_print_output({"image_path": path})
    except Exception as e:
        if is_gated_repo_error(e): soft_skip("gated model (no access/auth)"); return
        if is_missing_model_error(e): soft_skip("model not found on Hugging Face"); return
        safe_print_output({"error": "image-to-image failed", "reason": repr(e),
                           "hint": "Ensure the model supports img2img/inpainting and diffusers is installed."})
