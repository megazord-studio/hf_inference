#!/usr/bin/env python3
"""
app/main.py

Entry point that:
- prints environment/tooling header
- loads demos from ./demo.yaml
- iterates demos and dispatches to the correct runner from app.runners.RUNNERS
- respects RESPECT_SKIP=1 to skip items marked `skipped: true` in demo.yaml

It also makes itself robust to being run as a file path (uv run app/main.py)
by inserting the repository root on sys.path before importing app.*.
"""

import os
import sys
from typing import Any, Dict

# --- ensure the project root is importable when running as a file path ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.helpers import (
    print_header,
    load_demo,
    device_str,
    safe_print_output,
)
from app.runners import RUNNERS


def _announce(model_id: str, task: str):
    print(f"=== {model_id} ({task}) ===")


def main():
    print_header()
    demos = load_demo("./demo.yaml")
    dev = device_str()
    respect_skip = os.getenv("RESPECT_SKIP", "0") == "1"

    # optional narrow-by filters via env
    only_task = os.getenv("ONLY_TASK", "").strip().lower()
    only_model = os.getenv("ONLY_MODEL", "").strip().lower()

    for item in demos:
        model_id = item.get("model_id", "")
        task = item.get("task", "")
        payload: Dict[str, Any] = item.get("payload", {}) or {}

        # env filters
        if only_task and task.lower() != only_task:
            continue
        if only_model and model_id.lower() != only_model:
            continue

        if respect_skip and item.get("skipped", False):
            _announce(model_id, task)
            print("— SKIPPED —")
            continue

        _announce(model_id, task)
        runner = RUNNERS.get(task)
        if not runner:
            safe_print_output({
                "error": "Could not run. Unsupported task.",
                "Reason": f"Unknown task '{task}'",
                "supported_tasks": sorted(RUNNERS.keys()),
            })
            continue

        spec = {"model_id": model_id, "task": task, "payload": payload}

        try:
            runner(spec, dev)
        except Exception as e:
            # Runners already implement detailed handling/fallbacks; this is the last-resort guard.
            safe_print_output({
                "error": f"{task} failed",
                "reason": repr(e),
                "hint": None,
                "traceback": [] if os.getenv("JAM_DEBUG", "0") != "1" else [str(e.__class__.__name__)],
            })


if __name__ == "__main__":
    # Make stdout line-buffered when piped
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    main()
