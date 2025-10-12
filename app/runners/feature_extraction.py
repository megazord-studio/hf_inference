import numpy as np
import torch
from transformers import pipeline, AutoProcessor, AutoModel
from app.helpers import device_arg, device_str, safe_print_output
from app.utilities import is_gated_repo_error, is_missing_model_error, soft_skip

def run_feature_extraction(spec, dev: str):
    text = spec["payload"]["prompt"]
    try:
        if "clip" in spec["model_id"].lower():
            proc = AutoProcessor.from_pretrained(spec["model_id"])
            model = AutoModel.from_pretrained(spec["model_id"]).to(device_str())
            inputs = proc(text=[text], return_tensors="pt", padding=True).to(model.device)
            with torch.inference_mode():
                out = model.get_text_features(**inputs)
            safe_print_output({"embedding_shape": tuple(out.shape)}); return
        pl = pipeline("feature-extraction", model=spec["model_id"], device=device_arg(dev))
        vec = pl(text)
        safe_print_output({"embedding_shape": np.array(vec).shape})
    except Exception as e:
        if is_gated_repo_error(e): soft_skip("gated model (no access/auth)"); return
        if is_missing_model_error(e): soft_skip("model not found on Hugging Face"); return
        safe_print_output({"error": "feature-extraction failed", "reason": repr(e)})
