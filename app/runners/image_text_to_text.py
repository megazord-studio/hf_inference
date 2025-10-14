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
    Returns the result as a dictionary instead of printing.
    """
    payload = spec["payload"]

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
    mid = spec["model_id"].lower()

    if "llava" in mid:
        return _vlm_llava(spec, img, prompt, dev)
    if "florence-2" in mid or "florence" in mid:
        return _vlm_florence2(spec, img, prompt, dev)
    if "minicpm" in mid or "cpm" in mid:
        return _vlm_minicpm(spec, img, prompt, dev)

    try:
        pl: ImageTextToTextPipeline = pipeline(
            task="image-text-to-text",
            model=spec["model_id"],
            trust_remote_code=True,
            device=device_arg(dev),
        )
        # Call with keyword arguments to satisfy typing stubs
        out_any: Any = pl(image=img, text=prompt)

        if isinstance(out_any, dict) and "text" in out_any:
            return {"text": out_any["text"]}
        if isinstance(out_any, list) and out_any:
            first = out_any[0]
            if isinstance(first, dict) and "generated_text" in first:
                return {"text": first["generated_text"]}
            if (
                isinstance(first, list)
                and first
                and isinstance(first[0], dict)
                and "generated_text" in first[0]
            ):
                return {"text": first[0]["generated_text"]}
        if isinstance(out_any, str):
            return {"text": out_any}
    except Exception:
        pass

    try:
        pl2: ImageToTextPipeline = pipeline(
            task="image-to-text",
            model=spec["model_id"],
            trust_remote_code=True,
            device=device_arg(dev),
        )
        out2 = pl2(img)  # list[dict[str, Any]]
        if (
            isinstance(out2, list)
            and out2
            and isinstance(out2[0], dict)
            and "generated_text" in out2[0]
        ):
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
