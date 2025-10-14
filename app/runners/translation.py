from typing import Any
from typing import Dict

from transformers import pipeline

from app.helpers import device_arg
from app.helpers import safe_json
from app.types import RunnerSpec
from app.utilities import is_gated_repo_error
from app.utilities import is_missing_model_error


def run_translation(spec: RunnerSpec, dev: str) -> Dict[str, Any]:
    """
    Run translation inference.
    Returns the result as a dictionary instead of printing.
    """
    p = spec["payload"]
    mid = spec["model_id"].lower()
    try:
        if "mbart" in mid:
            src, tgt = p.get("src_lang"), p.get("tgt_lang")
            if not src or not tgt:
                return {
                    "error": "translation failed",
                    "reason": "facebook/mbart-large-50-many-to-many-mmt requires src_lang and tgt_lang.",
                    "hint": "Add src_lang/tgt_lang to this demo item in demo.yaml.",
                    "example": {"src_lang": "en_XX", "tgt_lang": "de_DE"},
                }
        pl = pipeline(
            "translation", model=spec["model_id"], device=device_arg(dev)
        )
        out = (
            pl(
                p["prompt"],
                src_lang=p.get("src_lang"),
                tgt_lang=p.get("tgt_lang"),
            )
            if ("src_lang" in p or "tgt_lang" in p)
            else pl(p["prompt"])
        )
        return safe_json(out)
    except Exception as e:
        if is_gated_repo_error(e):
            return {"skipped": True, "reason": "gated model (no access/auth)"}
        if is_missing_model_error(e):
            return {
                "skipped": True,
                "reason": "model not found on Hugging Face",
            }
        return {"error": "translation failed", "reason": repr(e)}
