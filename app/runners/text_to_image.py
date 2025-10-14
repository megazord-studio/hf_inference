import importlib
from typing import Any
from typing import Dict
from typing import Optional
from typing import Type

import torch
from diffusers import StableDiffusionPipeline

from app.helpers import device_str
from app.helpers import image_to_bytes
from app.runners.patches.patch_offline_kwarg import _patch_offload_kwarg
from app.types import RunnerSpec
from app.utilities import is_gated_repo_error
from app.utilities import is_missing_model_error

# Safe runtime lookup; avoids assigning to an imported type
AutoT2I: Optional[Type[Any]]
try:
    _diffusers = importlib.import_module("diffusers")
    AutoT2I = getattr(_diffusers, "AutoPipelineForText2Image")
except Exception:
    AutoT2I = None


def _choose_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.float16
    if (
        getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
    ):
        return torch.float16
    return torch.float32


_patch_offload_kwarg()


def run_text_to_image(spec: RunnerSpec, dev: str) -> Dict[str, Any]:
    """
    Run text-to-image inference.
    Returns image as bytes in a dictionary with metadata.
    """
    prompt = spec["payload"]["prompt"]
    model_id = spec["model_id"]

    try:
        dtype = _choose_dtype()
        common_kwargs = dict(
            torch_dtype=dtype,
            use_safetensors=True,
            low_cpu_mem_usage=False,  # avoid init-empty-weights path
            device_map=None,  # avoid auto device mapping at load time
            trust_remote_code=False,
        )

        # Help mypy for the runtime alias
        AutoT2I_t: Any = AutoT2I

        if AutoT2I_t is not None:
            try:
                pipe = AutoT2I_t.from_pretrained(
                    model_id,
                    safety_checker=None,
                    feature_extractor=None,
                    **common_kwargs,
                )
            except Exception:
                pipe = AutoT2I_t.from_pretrained(
                    model_id,
                    **common_kwargs,
                )
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                safety_checker=None,
                feature_extractor=None,
                **common_kwargs,
            )

        pipe = pipe.to(device_str())

        with torch.inference_mode():
            result = pipe(prompt=prompt, num_inference_steps=20)
            img = result.images[0]

        img_bytes = image_to_bytes(img, img_format="PNG")
        return {
            "file_data": img_bytes,
            "file_name": f"sd_{model_id.replace('/', '_')}.png",
            "content_type": "image/png",
        }

    except Exception as e:
        if is_gated_repo_error(e):
            return {
                "skipped": True,
                "reason": "gated model (no access/auth)",
                "hint": "Use runwayml/stable-diffusion-v1-5 or sdxl-turbo.",
            }
        if is_missing_model_error(e):
            return {
                "skipped": True,
                "reason": "model not found on Hugging Face",
            }
        return {"error": "text-to-image failed", "reason": repr(e)}
