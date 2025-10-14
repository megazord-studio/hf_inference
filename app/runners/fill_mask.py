from typing import Any
from typing import Dict
from typing import Tuple

from transformers import AutoTokenizer
from transformers import pipeline

from app.helpers import device_arg
from app.helpers import safe_json
from app.types import RunnerSpec
from app.utilities import is_gated_repo_error
from app.utilities import is_missing_model_error


def _normalize_mask_sentence(model_id: str, sentence: str) -> Tuple[str, str]:
    tok = AutoTokenizer.from_pretrained(model_id)  # nosec
    mask_token = tok.mask_token or "<mask>"
    s = (
        (sentence or "")
        .replace("<mask>", mask_token)
        .replace("[MASK]", mask_token)
        .replace("[mask]", mask_token)
    )
    if mask_token not in s:
        s = f"The capital of Switzerland is {mask_token}."
    parts = s.split(mask_token)
    if len(parts) > 2:
        s = mask_token.join([parts[0], parts[1]])
    return s, mask_token


def run_fill_mask(spec: RunnerSpec, dev: str) -> Dict[str, Any]:
    """
    Run fill mask inference.
    Returns the result as a dictionary instead of printing.
    """
    try:
        pl = pipeline(
            "fill-mask", model=spec["model_id"], device=device_arg(dev)
        )
        p = spec["payload"]
        s1, _ = _normalize_mask_sentence(
            spec["model_id"], p.get("mask_sentence", "")
        )
        s2, _ = _normalize_mask_sentence(
            spec["model_id"], p.get("mask_sentence_alt", "")
        )
        r1, r2 = pl(s1), pl(s2)
        return safe_json({"result_1": r1, "result_2": r2})
    except Exception as e:
        if is_gated_repo_error(e):
            return {"skipped": True, "reason": "gated model (no access/auth)"}
        if is_missing_model_error(e):
            return {
                "skipped": True,
                "reason": "model not found on Hugging Face",
            }
        return {"error": "fill-mask failed", "reason": repr(e)}
