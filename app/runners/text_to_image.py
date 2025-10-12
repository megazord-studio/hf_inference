import os
import torch
from diffusers import StableDiffusionPipeline
try:
    from diffusers import AutoPipelineForText2Image
except Exception:
    AutoPipelineForText2Image = None
from app.helpers import safe_print_output
from app.utilities import is_gated_repo_error, is_missing_model_error, soft_skip
from app.helpers import OUT_DIR, device_str

def run_text_to_image(spec, dev: str):
    prompt = spec["payload"]["prompt"]
    try:
        if AutoPipelineForText2Image is not None:
            try:
                pipe = AutoPipelineForText2Image.from_pretrained(
                    spec["model_id"], torch_dtype=torch.float16, variant="fp16"
                ).to(device_str())
            except Exception:
                pipe = AutoPipelineForText2Image.from_pretrained(
                    spec["model_id"], torch_dtype=torch.float16
                ).to(device_str())
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                spec["model_id"], torch_dtype=torch.float16
            ).to(device_str())
        with torch.inference_mode():
            img = pipe(prompt=prompt, num_inference_steps=20).images[0]
        path = os.path.join(OUT_DIR, f"sd_{spec['model_id'].replace('/', '_')}.png")
        img.save(path)
        safe_print_output({"image_path": path})
    except Exception as e:
        if is_gated_repo_error(e): soft_skip("gated model (no access/auth)", "Use runwayml/stable-diffusion-v1-5 or sdxl-turbo."); return
        if is_missing_model_error(e): soft_skip("model not found on Hugging Face"); return
        safe_print_output({"error": "text-to-image failed", "reason": repr(e)})
