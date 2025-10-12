import os
import torch
from PIL import Image as PILImage
from transformers import pipeline
from app.helpers import device_arg, ensure_image, safe_print_output, OUT_DIR
from app.utilities import is_gated_repo_error, is_missing_model_error, soft_skip

def run_depth_estimation(spec, dev: str):
    img = ensure_image(spec["payload"]["image_path"])
    try:
        pl = pipeline("depth-estimation", model=spec["model_id"], device=device_arg(dev))
        out = pl(img)
        depth = out.get("predicted_depth", None) if isinstance(out, dict) else None
        if isinstance(depth, torch.Tensor):
            d = depth.detach().cpu().numpy()
            d = (d - d.min()); d = d / (d.max() if d.max() > 0 else 1)
            d_img = PILImage.fromarray((d * 255).astype("uint8"))
            path = os.path.join(OUT_DIR, f"depth_{spec['model_id'].replace('/', '_')}.png")
            d_img.save(path)
            safe_print_output({"depth_path": path})
        else:
            safe_print_output(out)
    except Exception as e:
        if is_gated_repo_error(e): soft_skip("gated model (no access/auth)"); return
        if is_missing_model_error(e): soft_skip("model not found on Hugging Face"); return
        safe_print_output({"error": "depth-estimation failed", "reason": repr(e)})
