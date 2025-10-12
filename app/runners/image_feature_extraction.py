import numpy as np
import torch
from transformers import pipeline, AutoProcessor, AutoModel
from app.helpers import device_arg, device_str, ensure_image, safe_print_output
from app.utilities import is_gated_repo_error, is_missing_model_error, soft_skip

def run_image_feature_extraction(spec, dev: str):
    img = ensure_image(spec["payload"]["image_path"])
    try:
        if "clip" in spec["model_id"].lower():
            proc = AutoProcessor.from_pretrained(spec["model_id"])
            model = AutoModel.from_pretrained(spec["model_id"]).to(device_str())
            inputs = proc(images=img, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                feats = model.get_image_features(**inputs)
            safe_print_output({"embedding_shape": tuple(feats.shape)}); return
        pl = pipeline("image-feature-extraction", model=spec["model_id"], device=device_arg(dev))
        feats = pl(img)
        safe_print_output({"embedding_shape": np.array(feats).shape})
    except Exception as e:
        if is_gated_repo_error(e): soft_skip("gated model (no access/auth)"); return
        if is_missing_model_error(e): soft_skip("model not found on Hugging Face"); return
        safe_print_output({"error": "image-feature-extraction failed", "reason": repr(e)})
