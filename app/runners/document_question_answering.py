from transformers import pipeline
from app.helpers import device_arg, ensure_image, safe_print_output
from app.utilities import is_gated_repo_error, is_missing_model_error, soft_hint_error, soft_skip

def run_doc_qa(spec, dev: str):
    img = ensure_image(spec["payload"]["image_path"])
    mid = spec["model_id"].lower()
    if "layoutlmv3" in mid:
        soft_skip("model incompatible with document-question-answering pipeline",
                  "Use a doc-VQA model like impira/layoutlm-document-qa or naver-clova-ix/donut-base-finetuned-docvqa.")
        return
    try:
        pl = pipeline("document-question-answering", model=spec["model_id"], device=device_arg(dev), trust_remote_code=True)
        out = pl(image=img, question=spec["payload"]["question"])
        safe_print_output(out)
    except Exception as e:
        if is_gated_repo_error(e):
            soft_skip("gated model (no access/auth)",
                      "Pick a published doc-VQA model, e.g., impira/layoutlm-document-qa or donut docvqa.")
            return
        if is_missing_model_error(e):
            soft_skip("model not found on Hugging Face",
                      "Pick a published doc-VQA model, e.g., impira/layoutlm-document-qa or donut docvqa.")
            return
        soft_hint_error("document-question-answering failed", repr(e),
                        "Ensure the model supports doc-qa or switch to a compatible doc-VQA model.")
