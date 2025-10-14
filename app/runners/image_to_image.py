from typing import Any
from typing import Dict

import torch
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image

from app.helpers import device_str
from app.helpers import ensure_image
from app.helpers import get_upload_file_image
from app.helpers import image_to_bytes
from app.runners.patches.patch_offline_kwarg import _patch_offload_kwarg
from app.types import RunnerSpec
from app.utilities import is_gated_repo_error
from app.utilities import is_missing_model_error

_patch_offload_kwarg()


def run_image_to_image(spec: RunnerSpec, dev: str) -> Dict[str, Any]:
    """
    Run image-to-image inference.
    Accepts either init_image_path or UploadFile from spec["files"]["image"].
    Returns generated image as bytes.
    """
    p = spec["payload"]

    # Handle UploadFile or fallback to path
    init_img = get_upload_file_image(spec.get("files", {}).get("image"))
    if init_img is None:
        init_img = ensure_image(p.get("init_image_path", "image.jpg"))

    model_id = spec["model_id"]
    try:
        if "inpaint" in model_id.lower():
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                model_id, torch_dtype=torch.float16
            ).to(device_str())
            w, h = init_img.size
            mask = Image.new("L", (w, h), 255)
            with torch.inference_mode():
                out = pipe(
                    prompt=p["prompt"],
                    image=init_img,
                    mask_image=mask,
                    guidance_scale=7.0,
                    num_inference_steps=25,
                )
            img = out.images[0]
        else:
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_id, torch_dtype=torch.float16
            ).to(device_str())
            with torch.inference_mode():
                out = pipe(
                    prompt=p["prompt"],
                    image=init_img,
                    strength=0.6,
                    guidance_scale=7.0,
                    num_inference_steps=25,
                )
            img = out.images[0]

        # Convert to bytes
        img_bytes = image_to_bytes(img, img_format="PNG")
        return {
            "file_data": img_bytes,
            "file_name": f"sd_img2img_{model_id.replace('/', '_')}.png",
            "content_type": "image/png",
        }
    except Exception as e:
        if is_gated_repo_error(e):
            return {"skipped": True, "reason": "gated model (no access/auth)"}
        if is_missing_model_error(e):
            return {
                "skipped": True,
                "reason": "model not found on Hugging Face",
            }
        return {
            "error": "image-to-image failed",
            "reason": repr(e),
            "hint": "Ensure the model supports img2img/inpainting and diffusers is installed.",
        }
