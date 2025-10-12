from transformers import pipeline
from app.helpers import device_arg, safe_print_output, to_dataframe
from app.utilities import is_gated_repo_error, is_missing_model_error, soft_skip

def run_table_qa(spec, dev: str):
    try:
        p = spec["payload"]
        df = to_dataframe(p["table"])
        dev_for_tapas = "cpu" if "tapas" in spec["model_id"].lower() else device_arg(dev)
        pl = pipeline("table-question-answering", model=spec["model_id"], device=dev_for_tapas)
        out = pl({"table": df, "query": p["table_query"]})
        safe_print_output(out)
    except Exception as e:
        if is_gated_repo_error(e): soft_skip("gated model (no access/auth)"); return
        if is_missing_model_error(e): soft_skip("model not found on Hugging Face"); return
        safe_print_output({"error": "table-question-answering failed", "reason": repr(e),
                           "hint": "If using TAPAS, try CPU; verify table format and query."})
