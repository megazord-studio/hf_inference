from typing import Any
from typing import Dict

from PIL.Image import Image as PILImage
from transformers import pipeline
from transformers.pipelines import ImageTextToTextPipeline
from transformers.pipelines import ImageToTextPipeline

from app.helpers import device_arg
from app.helpers import ensure_image
from app.helpers import get_upload_file_image
from app.types import RunnerSpec
from app.utilities import _final_caption_fallback
from app.utilities import _vlm_florence2
from app.utilities import _vlm_llava
from app.utilities import _vlm_minicpm


def run_vlm_image_text_to_text(spec: RunnerSpec, dev: str) -> Dict[str, Any]:
    """
    Run vision-language model image-text-to-text inference.
    Accepts either image_path or UploadFile from spec["files"]["image"].
    Returns the result as a dictionary.

    IMPORTANT:
    - extra_args are passed DIRECTLY to the HF pipeline constructor (pipeline(..., **extra_args)).
    - No special handling of max_tokens / max_new_tokens.
    - No call-time generation kwargs are injected; only pipeline construction is configured via extra_args.
    """
    payload: Dict[str, Any] = spec.get("payload", {}) or {}
    extra_args: Dict[str, Any] = spec.get("extra_args", {}) or {}

    # Handle UploadFile or fallback to path
    img: PILImage | None = get_upload_file_image(
        spec.get("files", {}).get("image")
    )
    if img is None:
        img = ensure_image(payload.get("image_path", "image.jpg"))
    if img is None:
        return {
            "error": "image-text-to-text failed",
            "reason": "invalid image",
        }

    prompt: str = str(payload.get("prompt", "Describe the image briefly."))
    mid = str(spec.get("model_id", "")).lower()

    # Model-specific helper paths (unchanged)
    if "llava" in mid:
        return _vlm_llava(spec, img, prompt, dev)
    if "florence-2" in mid or "florence" in mid:
        return _vlm_florence2(spec, img, prompt, dev)
    if "minicpm" in mid or "cpm" in mid:
        return _vlm_minicpm(spec, img, prompt, dev)

    # Generic path 1: image-text-to-text
    try:
        pl: ImageTextToTextPipeline = pipeline(
            task="image-text-to-text",
            model=spec["model_id"],
            trust_remote_code=True,
            device=device_arg(dev),
            **extra_args,  # pass through exactly as provided
        )
        out_any: Any = pl(image=img, text=prompt)
        text = _unwrap_text(out_any)
        if text:
            return {"text": text}
    except Exception:
        pass

    # Generic path 2: image-to-text fallback
    try:
        pl2: ImageToTextPipeline = pipeline(
            task="image-to-text",
            model=spec["model_id"],
            trust_remote_code=True,
            device=device_arg(dev),
            **extra_args,  # pass through exactly as provided
        )
        out2: Any = pl2(img)
        text = _unwrap_text(out2)
        if text:
            return {"text": text}
    except Exception:
        pass

    cap = _final_caption_fallback(img, dev)
    if isinstance(cap, dict) and "text" in cap:
        return {"text": cap["text"], "note": "fallback caption used"}
    return cap


def _unwrap_text(out_any: Any) -> str | None:
    if isinstance(out_any, dict) and "text" in out_any:
        return str(out_any["text"])
    if isinstance(out_any, list) and out_any:
        first = out_any[0]
        if isinstance(first, dict):
            if "generated_text" in first:
                return str(first["generated_text"])
            if "text" in first:
                return str(first["text"])
        if (
            isinstance(first, list)
            and first
            and isinstance(first[0], dict)
            and "generated_text" in first[0]
        ):
            return str(first[0]["generated_text"])
    if isinstance(out_any, str):
        return out_any
    return None
