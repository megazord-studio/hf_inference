import importlib
from typing import Any
from typing import Dict
from typing import Optional
from typing import Type

import torch
from diffusers import StableDiffusionPipeline
from transformers import pipeline

from app.helpers import device_arg
from app.helpers import device_str
from app.helpers import image_to_bytes
from app.runners.patches.patch_offline_kwarg import _patch_offload_kwarg
from app.types import RunnerSpec
from app.utilities import is_gated_repo_error
from app.utilities import is_missing_model_error
from app.utilities import is_missing_model_index_error

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
            trust_remote_code=True,  # Enable trust_remote_code for custom models
        )

        # StableDiffusionPipeline doesn't support trust_remote_code
        sd_kwargs = {
            k: v for k, v in common_kwargs.items() if k != "trust_remote_code"
        }

        # Help mypy for the runtime alias
        AutoT2I_t: Any = AutoT2I

        pipe = None

        # Try AutoPipelineForText2Image first
        if AutoT2I_t is not None:
            try:
                pipe = AutoT2I_t.from_pretrained(
                    model_id,
                    safety_checker=None,
                    feature_extractor=None,
                    **common_kwargs,
                )
            except Exception as e:
                # If model_index.json is missing, try without safety_checker args
                if is_missing_model_index_error(e):
                    try:
                        pipe = AutoT2I_t.from_pretrained(
                            model_id,
                            **common_kwargs,
                        )
                    except Exception:
                        # Will fall through to next fallback
                        pass
                else:
                    # Try without safety_checker args for other errors
                    try:
                        pipe = AutoT2I_t.from_pretrained(
                            model_id,
                            **common_kwargs,
                        )
                    except Exception:
                        # Will fall through to next fallback
                        pass

        # Fallback to StableDiffusionPipeline if AutoPipelineForText2Image failed
        if pipe is None:
            try:
                pipe = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    safety_checker=None,
                    feature_extractor=None,
                    **sd_kwargs,
                )
            except Exception as e:
                # If model_index.json is missing, try transformers pipeline
                if is_missing_model_index_error(e):
                    try:
                        # Use transformers pipeline as final fallback
                        pl = pipeline(
                            "text-to-image",
                            model=model_id,
                            device=device_arg(dev),
                            trust_remote_code=True,
                        )
                        # Generate image
                        result = pl(prompt)
                        # Handle different output formats
                        if isinstance(result, list) and len(result) > 0:
                            img = result[0]
                        else:
                            img = result

                        img_bytes = image_to_bytes(img, img_format="PNG")
                        return {
                            "file_data": img_bytes,
                            "file_name": f"t2i_{model_id.replace('/', '_')}.png",
                            "content_type": "image/png",
                        }
                    except Exception:
                        # Re-raise the original error if transformers pipeline also fails
                        raise e
                else:
                    raise

        # If we got a pipe object, use it for inference
        if pipe is not None:
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

        # If no pipe was created, return an error
        return {
            "error": "text-to-image failed",
            "reason": "Could not load model with any available method",
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
