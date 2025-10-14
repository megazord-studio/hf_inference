from typing import Any
from typing import Dict
from typing import cast

from transformers import AutoTokenizer
from transformers import pipeline
from transformers.pipelines import TableQuestionAnsweringPipeline

from app.helpers import device_arg
from app.helpers import safe_json
from app.types import RunnerSpec
from app.utilities import is_gated_repo_error
from app.utilities import is_missing_model_error


def _build_tapas_dataframe(table: Any) -> Any:
    import pandas as pd

    if isinstance(table, list) and table and isinstance(table[0], list):
        headers = [str(h) for h in table[0]]
        rows = [[str(c) for c in r] for r in table[1:]]
        if not rows:
            rows = [[""] * len(headers)]
        df = pd.DataFrame(rows, columns=headers)
    else:
        from app.helpers import to_dataframe  # lazy import to avoid cycles

        df = to_dataframe(table)

    df = df.fillna("").astype(str).reset_index(drop=True)
    return df


def run_table_qa(spec: RunnerSpec, dev: str) -> Dict[str, Any]:
    df: Any = None
    query: str = ""
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

        force_cpu = -1 if "tapas" in model_id.lower() else device_arg(dev)

        tokenizer = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        except Exception:
            pass

        qa_kwargs: Dict[str, Any] = dict(model=model_id, device=force_cpu)
        if tokenizer is not None:
            qa_kwargs["tokenizer"] = tokenizer

        pl = cast(
            TableQuestionAnsweringPipeline,
            pipeline("table-question-answering", **qa_kwargs),
        )

        out = pl({"table": df, "query": query})
        return safe_json(out)

    except Exception as e:
        msg = repr(e)
        if (
            ("Categorical(logits" in msg or "nan" in msg.lower())
            and df is not None
            and query
        ):
            try:
                pl = cast(
                    TableQuestionAnsweringPipeline,
                    pipeline(
                        "table-question-answering",
                        model=spec["model_id"],
                        device=-1,
                    ),
                )
                out = pl({"table": df, "query": query})
                return safe_json(out)
            except Exception as e2:
                e = e2

        if is_gated_repo_error(e):
            return {"skipped": True, "reason": "gated model (no access/auth)"}
        if is_missing_model_error(e):
            return {
                "skipped": True,
                "reason": "model not found on Hugging Face",
            }
        return {"error": "table-question-answering failed", "reason": repr(e)}
