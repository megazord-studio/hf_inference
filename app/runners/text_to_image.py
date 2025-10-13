import torch
from diffusers import StableDiffusionPipeline
try:
    from diffusers import AutoPipelineForText2Image
except Exception:
    AutoPipelineForText2Image = None
from app.helpers import image_to_bytes
from app.utilities import is_gated_repo_error, is_missing_model_error
from app.helpers import device_str

def run_text_to_image(spec, dev: str):
    """
    Run text-to-image inference.
    Returns image as bytes in a dictionary with metadata.
    """
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
        
        # Convert image to bytes
        img_bytes = image_to_bytes(img, format="PNG")
        return {
            "file_data": img_bytes,
            "file_name": f"sd_{spec['model_id'].replace('/', '_')}.png",
            "content_type": "image/png"
        }
    except Exception as e:
        if is_gated_repo_error(e):
            return {"skipped": True, "reason": "gated model (no access/auth)", "hint": "Use runwayml/stable-diffusion-v1-5 or sdxl-turbo."}
        if is_missing_model_error(e):
            return {"skipped": True, "reason": "model not found on Hugging Face"}
        return {"error": "text-to-image failed", "reason": repr(e)}
