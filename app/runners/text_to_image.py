import importlib
from typing import Any
from typing import Dict
from typing import Optional
from typing import Type
from typing import cast

import torch
from diffusers import StableDiffusionPipeline
from huggingface_hub import HfApi

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


def _repo_has_diffusers_index(model_id: str) -> bool:
    """Check if the repo exposes a Diffusers pipeline (model_index.json)."""
    try:
        api = HfApi()
        return api.file_exists(model_id, "model_index.json", repo_type="model")
    except Exception:
        # If we can't check, default to True so we try Diffusers first
        return True


def _repo_has_safetensors(model_id: str) -> bool:
    """Return True if the repo contains any .safetensors weights.

    Some Diffusers repos only ship .bin weights. Forcing use_safetensors=True
    will fail to load those. Detect presence and only enable safetensors when available.
    """
    try:
        api = HfApi()
        files = api.list_repo_files(model_id, repo_type="model")
        return any(f.endswith(".safetensors") for f in files)
    except Exception:
        # Be conservative: prefer allowing .bin weights
        return False


def _run_via_transformers(
    model_id: str,
    prompt: str,
    dev: str,
    extra_args: Dict[str, Any],
) -> Any:
    """Run text-to-image via Transformers pipeline with trust_remote_code.

    Do not pass a task string; let Transformers infer from the repo config
    (pipeline_tag) to avoid KeyError for unknown tasks like 'text-to-image'.
    """
    from transformers import pipeline as hf_pipeline
    from app.helpers import device_arg
    from PIL import Image as _Image

    hf_pipe_any = cast(Any, hf_pipeline)
    # Let transformers infer the task from the repo (requires trust_remote_code)
    pl = hf_pipe_any(
        model=model_id,
        device=device_arg(dev),
        trust_remote_code=True,
        **extra_args,
    )

    with torch.inference_mode():
        out = pl(prompt=prompt)

    # Normalize output to a PIL Image
    if isinstance(out, _Image.Image):
        return out
    if isinstance(out, dict):
        if out.get("images"):
            return out["images"][0]
        if "image" in out:
            return out["image"]
        raise RuntimeError(
            "Transformers text-to-image returned unsupported dict format"
        )
    if isinstance(out, list) and out:
        first = out[0]
        if isinstance(first, _Image.Image):
            return first
        if isinstance(first, dict) and "image" in first:
            return first["image"]
        raise RuntimeError(
            "Transformers text-to-image returned unsupported list format"
        )
    raise RuntimeError(
        f"Unexpected transformers text-to-image output type: {type(out)}"
    )


_patch_offload_kwarg()


def run_text_to_image(spec: RunnerSpec, dev: str) -> Dict[str, Any]:
    """
    Run text-to-image inference.
    Returns image as bytes in a dictionary with metadata.
    """
    extra_args: Dict[str, Any] = spec.get("extra_args", {}) or {}

    prompt = spec["payload"]["prompt"]
    model_id = spec["model_id"]

    try:
        dtype = _choose_dtype()

        # Enable safetensors only when the repo actually provides them
        diffusers_repo = _repo_has_diffusers_index(model_id)
        has_safetensors = _repo_has_safetensors(model_id) if diffusers_repo else False

        common_kwargs = dict(
            torch_dtype=dtype,
            use_safetensors=has_safetensors,
            low_cpu_mem_usage=False,  # avoid init-empty-weights path
            device_map=None,  # avoid auto device mapping at load time
            trust_remote_code=False,
            **extra_args,
        )

        # Help mypy for the runtime alias
        AutoT2I_t: Any = AutoT2I

        img: Any = None

        # If repo clearly isn't Diffusers, go straight to Transformers
        if not diffusers_repo:
            img = _run_via_transformers(model_id, prompt, dev, extra_args)
        else:
            # Try loading with Diffusers first
            try:
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
            except Exception as diffusers_err:
                # Fallback: try Transformers pipeline with trust_remote_code
                try:
                    img = _run_via_transformers(
                        model_id, prompt, dev, extra_args
                    )
                except Exception as transformers_err:
                    # Report both errors to aid debugging instead of hiding the fallback error
                    raise RuntimeError(
                        f"Diffusers load failed: {diffusers_err}; Transformers fallback failed: {transformers_err}"
                    )

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
        # Include a hint if the error mentions model_index.json (non-Diffusers repo)
        hint = None
        if "model_index.json" in str(e):
            hint = (
                "Repo is not a Diffusers model; ensure transformers fallback is allowed (trust_remote_code) and token has access."
            )
        out: Dict[str, Any] = {"error": "text-to-image failed", "reason": repr(e)}
        if hint:
            out["hint"] = hint
        return out
