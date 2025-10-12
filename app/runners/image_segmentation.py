import os
import numpy as np
import torch
from PIL import Image
from transformers import pipeline
from app.helpers import device_arg, ensure_image, safe_print_output, OUT_DIR
from app.utilities import is_gated_repo_error, is_missing_model_error, soft_skip

def _save_panoptic_masks(seg_list, prefix: str):
    saved = []
    for i, seg in enumerate(seg_list):
        mask = seg.get("mask", None)
        entry = {}
        for k, v in seg.items():
            if k == "mask": continue
            if k == "score":
                try: entry[k] = float(v)
                except Exception: entry[k] = v
            else:
                entry[k] = v
        if isinstance(mask, (Image.Image, np.ndarray, torch.Tensor)):
            if isinstance(mask, torch.Tensor):
                mask = mask.detach().cpu().numpy()
            if isinstance(mask, np.ndarray):
                from PIL import Image as PILImage
                mask_img = PILImage.fromarray((mask * 255).astype(np.uint8)) if mask.ndim == 2 else PILImage.fromarray(mask)
            else:
                mask_img = mask
            path = os.path.join(OUT_DIR, f"{prefix}_{i}.png")
            mask_img.save(path)
            entry["mask_path"] = path
        saved.append(entry)
    return saved

def run_image_segmentation(spec, dev: str):
    img = ensure_image(spec["payload"]["image_path"])
    try:
        pl = pipeline("image-segmentation", model=spec["model_id"], device=device_arg(dev))
        out = pl(img)
        saved = _save_panoptic_masks(out, f"seg_{spec['model_id'].replace('/', '_')}")
        safe_print_output(saved)
    except Exception as e:
        if is_gated_repo_error(e): soft_skip("gated model (no access/auth)"); return
        if is_missing_model_error(e): soft_skip("model not found on Hugging Face"); return
        safe_print_output({"error": "image-segmentation failed", "reason": repr(e)})
