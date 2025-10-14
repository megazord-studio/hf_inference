from typing import Any
from typing import Dict

from transformers import AutoTokenizer
from transformers import pipeline

from app.helpers import device_arg
from app.helpers import safe_json
from app.utilities import is_gated_repo_error
from app.utilities import is_missing_model_error


def _build_tapas_dataframe(table: Any) -> Any:
    """
    Build a pandas.DataFrame for TAPAS:
    - First row is treated as header
    - All values coerced to string
    - No NaNs
    - At least one data row
    """
    import pandas as pd

    # If caller passed a 2D list, interpret first row as header
    if isinstance(table, list) and table and isinstance(table[0], list):
        headers = [str(h) for h in table[0]]
        rows = [[str(c) for c in r] for r in table[1:]]
        # Ensure at least one data row (TAPAS hates empty tables)
        if not rows:
            rows = [[""] * len(headers)]
        df = pd.DataFrame(rows, columns=headers)
    else:
        # Fallback to existing helper, then sanitize
        from app.helpers import to_dataframe  # lazy import to avoid cycles

        df = to_dataframe(table)

    # Sanitize: fill NaNs, coerce to string, reset index
    df = df.fillna("").astype(str).reset_index(drop=True)
    return df


def run_table_qa(spec: Any, dev: str) -> Dict[str, Any]:
    """
    Run table question answering inference.
    Returns the result as a dictionary instead of printing.
    """
    try:
        p = spec["payload"]
        model_id = spec["model_id"]
        query = str(p.get("table_query", "")).strip()
        if not query:
            return {
                "error": "table-question-answering failed",
                "reason": "empty query",
            }

        df = _build_tapas_dataframe(p["table"])

        # TAPAS must run on CPU for stability; transformers expects -1 for CPU
        force_cpu = -1 if "tapas" in model_id.lower() else device_arg(dev)

        # Some combos behave better when tokenizer is passed explicitly
        tokenizer = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        except Exception:
            pass

        qa_kwargs = dict(model=model_id, device=force_cpu)
        if tokenizer is not None:
            qa_kwargs["tokenizer"] = tokenizer

        pl = pipeline("table-question-answering", **qa_kwargs)

        # Run
        out = pl({"table": df, "query": query})
        return safe_json(out)

    except Exception as e:
        # Retry once with stricter CPU + sanitized fallback if it looks like the logits/NaN failure
        msg = repr(e)
        if "Categorical(logits" in msg or "nan" in msg.lower():
            try:
                # Rebuild the table even more defensively (already sanitized above, so just re-run on CPU)
                pl = pipeline(
                    "table-question-answering",
                    model=spec["model_id"],
                    device=-1,  # hard CPU
                )
                out = pl({"table": df, "query": query})
                return safe_json(out)
            except Exception as e2:
                e = e2  # fall through to standard error handling

        if is_gated_repo_error(e):
            return {"skipped": True, "reason": "gated model (no access/auth)"}
        if is_missing_model_error(e):
            return {
                "skipped": True,
                "reason": "model not found on Hugging Face",
            }
        return {
            "error": "table-question-answering failed",
            "reason": repr(e),
        }
