from transformers import pipeline
from app.helpers import device_arg, safe_print_output
from app.utilities import is_gated_repo_error, is_missing_model_error, soft_skip

def run_translation(spec, dev: str):
    p = spec["payload"]; mid = spec["model_id"].lower()
    try:
        if "mbart" in mid:
            src, tgt = p.get("src_lang"), p.get("tgt_lang")
            if not src or not tgt:
                safe_print_output({
                    "error": "translation failed",
                    "reason": "facebook/mbart-large-50-many-to-many-mmt requires src_lang and tgt_lang.",
                    "hint": "Add src_lang/tgt_lang to this demo item in demo.yaml.",
                    "example": {"src_lang": "en_XX", "tgt_lang": "de_DE"}
                })
                return
        pl = pipeline("translation", model=spec["model_id"], device=device_arg(dev))
        out = pl(p["prompt"], src_lang=p.get("src_lang"), tgt_lang=p.get("tgt_lang")) if ("src_lang" in p or "tgt_lang" in p) else pl(p["prompt"])
        safe_print_output(out)
    except Exception as e:
        if is_gated_repo_error(e): soft_skip("gated model (no access/auth)"); return
        if is_missing_model_error(e): soft_skip("model not found on Hugging Face"); return
        safe_print_output({"error": "translation failed", "reason": repr(e)})
