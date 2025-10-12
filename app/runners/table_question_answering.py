from transformers import pipeline
from app.helpers import device_arg, safe_json, to_dataframe
from app.utilities import is_gated_repo_error, is_missing_model_error

def run_table_qa(spec, dev: str):
    """
    Run table question answering inference.
    Returns the result as a dictionary instead of printing.
    """
    try:
        p = spec["payload"]
        df = to_dataframe(p["table"])
        dev_for_tapas = "cpu" if "tapas" in spec["model_id"].lower() else device_arg(dev)
        pl = pipeline("table-question-answering", model=spec["model_id"], device=dev_for_tapas)
        out = pl({"table": df, "query": p["table_query"]})
        return safe_json(out)
    except Exception as e:
        if is_gated_repo_error(e):
            return {"skipped": True, "reason": "gated model (no access/auth)"}
        if is_missing_model_error(e):
            return {"skipped": True, "reason": "model not found on Hugging Face"}
        return {"error": "table-question-answering failed", "reason": repr(e),
                "hint": "If using TAPAS, try CPU; verify table format and query."}
