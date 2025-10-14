from typing import Any
from typing import Dict

import numpy as np
import torch
from transformers import AutoModel
from transformers import AutoProcessor
from transformers import pipeline

from app.helpers import device_arg
from app.helpers import device_str
from app.types import RunnerSpec
from app.utilities import is_gated_repo_error
from app.utilities import is_missing_model_error


def run_feature_extraction(spec: RunnerSpec, dev: str) -> Dict[str, Any]:
    """
    Run feature extraction inference.
    Returns the result as a dictionary instead of printing.
    """
    text = spec["payload"]["prompt"]
    try:
        if "clip" in spec["model_id"].lower():
            proc = AutoProcessor.from_pretrained(spec["model_id"])  # nosec
            model = AutoModel.from_pretrained(spec["model_id"]).to(  # nosec
                device_str()
            )
            inputs = proc(text=[text], return_tensors="pt", padding=True).to(
                model.device
            )
            with torch.inference_mode():
                out = model.get_text_features(**inputs)
            return {"embedding_shape": tuple(out.shape)}
        pl = pipeline(
            "feature-extraction",
            model=spec["model_id"],
            device=device_arg(dev),
        )
        vec = pl(text)
        return {"embedding_shape": np.array(vec).shape}
    except Exception as e:
        if is_gated_repo_error(e):
            return {"skipped": True, "reason": "gated model (no access/auth)"}
        if is_missing_model_error(e):
            return {
                "skipped": True,
                "reason": "model not found on Hugging Face",
            }
        return {"error": "feature-extraction failed", "reason": repr(e)}
