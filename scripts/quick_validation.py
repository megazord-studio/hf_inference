"""Quick validation script to mirror frontend model filtering and scoring.

Usage:
    python scripts/quick_validation.py

Behavior:
    - Fetches preloaded models from the running backend (`/api/models/preloaded`).
  - Reconstructs task modality mappings (mirrors frontend `tasksCatalog.ts` + `taskModalities.ts`).
  - Aggregates models into unique (input modalities, output modalities) combinations.
  - Computes a heuristic trending metric (since backend does not expose `trendingScore`).
  - Ranks models per combination with the provided score formula:

    score = 0.4 * trending_norm + 0.35 * downloads_norm + 0.25 * likes_norm

    where:
      trending = normalized(likes / (downloads + 1))  (scaled to [0,1])
      log() is natural logarithm, with safe fallbacks for zero.
    - Prints the top 10 models for each combination.

Modes:
    MODE=goal-mode (default): prints top N models (10) per (input modalities, output modalities) combination.
    MODE=task-mode: prints top 2 models per individual HF task (pipeline_tag) using the same normalized score.

Notes:
    - The frontend expects a `trendingScore` from the server; none exists in current backend code.
        This script derives a lightweight proxy so the composite score is meaningful.
        - All metrics (trending, downloads, likes) normalized across the full fetched model set.
            Downloads & likes use log transform before min-max normalization to compress heavy tails.
        - Gated/private models are excluded from scoring and execution.
  - Output modality list is constrained to the frontend's selectable set.
  - Tasks whose input is 'various' do NOT support simultaneous multi-input combos (matching TS logic).
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
import os
import base64
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple
from datetime import datetime
import shutil
import time

TIMEOUT = 1200

try:  # Prefer httpx if available (in dev extras); fallback to urllib
    import httpx  # type: ignore
except Exception:  # pragma: no cover
    httpx = None  # type: ignore
    import json
    from urllib.request import urlopen
    from urllib.error import URLError

    def _http_get(url: str):
        try:
            with urlopen(url, timeout=TIMEOUT) as resp:  # type: ignore
                return json.loads(resp.read().decode("utf-8"))
        except URLError as e:  # pragma: no cover
            raise RuntimeError(f"GET {url} failed: {e}")
else:
    def _http_get(url: str):  # pragma: no cover - delegated to httpx
        with httpx.Client(timeout=TIMEOUT) as client:
            r = client.get(url)
            r.raise_for_status()
            return r.json()


# Frontend canonical modality sets
ALL_INPUT_MODALITIES: Sequence[str] = ["text", "image", "audio", "video", "document"]
ALL_OUTPUT_MODALITIES: Sequence[str] = [
    "text",
    "image",
    "audio",
    "video",
    "embedding",
    "boxes",
    "mask",
    "depth",
    "3d",
]


@dataclass(frozen=True)
class TaskInfo:
    id: str
    input: str | None
    output: str | None


# Minimal reconstruction of tasksInfo (only fields we need: id, input, output)
TASKS_INFO: List[TaskInfo] = [
    TaskInfo("text-generation", "text", "text"),
    TaskInfo("summarization", "text", "text"),
    TaskInfo("translation", "text", "text"),
    TaskInfo("question-answering", "text", "text"),
    TaskInfo("table-question-answering", "text", "text"),
    TaskInfo("text-classification", "text", "labels"),
    TaskInfo("token-classification", "text", "tags"),
    TaskInfo("zero-shot-classification", "text", "labels"),
    TaskInfo("fill-mask", "text", "text"),
    TaskInfo("text-ranking", "text", "scores"),
    TaskInfo("sentence-similarity", "text", "embedding"),
    TaskInfo("feature-extraction", "text", "embedding"),
    TaskInfo("image-classification", "image", "labels"),
    TaskInfo("object-detection", "image", "boxes"),
    TaskInfo("image-segmentation", "image", "mask"),
    TaskInfo("zero-shot-image-classification", "image", "labels"),
    TaskInfo("zero-shot-object-detection", "image", "boxes"),
    TaskInfo("keypoint-detection", "image", "points"),
    TaskInfo("depth-estimation", "image", "depth map"),
    TaskInfo("image-to-text", "image", "text"),
    TaskInfo("image-text-to-text", "image+text", "text"),
    TaskInfo("text-to-image", "text", "image"),
    TaskInfo("image-to-image", "image", "image"),
    TaskInfo("image-super-resolution", "image", "image"),
    TaskInfo("image-restoration", "image", "image"),
    TaskInfo("image-to-3d", "image", "3D"),
    TaskInfo("text-to-3d", "text", "3D"),
    TaskInfo("text-to-video", "text", "video"),
    TaskInfo("image-to-video", "image", "video"),
    TaskInfo("automatic-speech-recognition", "audio", "text"),
    TaskInfo("text-to-speech", "text", "audio"),
    TaskInfo("audio-classification", "audio", "labels"),
    TaskInfo("audio-to-audio", "audio", "audio"),
    TaskInfo("text-to-audio", "text", "audio"),
    TaskInfo("audio-text-to-text", "audio", "text"),
    TaskInfo("voice-activity-detection", "audio", "segments"),
    TaskInfo("time-series-forecasting", "series", "series"),
    TaskInfo("visual-document-retrieval", "image+text", "docs"),
    TaskInfo("any-to-any", "various", "various"),
]


@dataclass
class TaskModalities:
    input: List[str]
    output: List[str]
    multi_input_support: bool


def _parse_task_modalities(info: TaskInfo) -> TaskModalities:
    inp_raw = (info.input or "").lower()
    inputs: List[str] = []
    multi = False
    if inp_raw:
        if "image+text" in inp_raw:
            inputs.extend(["image", "text"])
            multi = True
        elif "various" in inp_raw:
            inputs.extend(["text", "image", "audio", "video", "document"])
            multi = False
        else:
            for m in ["text", "image", "audio", "video", "document"]:
                if m in inp_raw:
                    inputs.append(m)
            if len(inputs) > 1:
                multi = True
    out_raw = (info.output or "").lower()
    outputs: List[str] = []
    if out_raw:
        # Add standardized output types (subset matches frontend filters)
        mapping_targets = [
            "text",
            "image",
            "audio",
            "video",
            "3d",
            "embedding",
            "mask",
            "boxes",
            "scores",
            "points",
            "segments",
            "depth",
        ]
        for t in mapping_targets:
            if t in out_raw:
                # normalize '3d' and 'depth map'
                if t == "depth" and "depth" in out_raw:
                    outputs.append("depth")
                elif t == "3d":
                    outputs.append("3d")
                else:
                    outputs.append(t)
        if "image+text" in out_raw:
            for add in ["image", "text"]:
                if add not in outputs:
                    outputs.append(add)
    # Fallback defaults identical to frontend logic
    if not inputs:
        inputs = ["text"]
    if not outputs:
        outputs = ["text"]
    # Restrict outputs to selectable frontend set for combination enumeration
    outputs = [o for o in outputs if o in ALL_OUTPUT_MODALITIES]
    return TaskModalities(inputs, outputs, multi)


def build_task_modalities() -> Dict[str, TaskModalities]:
    mapping: Dict[str, TaskModalities] = {}
    for t in TASKS_INFO:
        mapping[t.id] = _parse_task_modalities(t)
    return mapping


def fetch_models(host: str) -> List[Dict[str, object]]:
    url = host.rstrip("/") + "/api/models/preloaded?limit=50000"
    data = _http_get(url)
    if not isinstance(data, list):  # pragma: no cover
        raise RuntimeError("Unexpected response shape for models list")
    return data  # Each item: { id, pipeline_tag, likes, downloads, ... }


def _minmax(values: Dict[str, float]) -> Dict[str, float]:
    if not values:
        return {}
    mn = min(values.values())
    mx = max(values.values())
    span = mx - mn if mx != mn else 1.0
    return {k: (v - mn) / span for k, v in values.items()}


def build_normalization_maps(models: Iterable[Dict[str, object]]) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Return normalized (trending, downloads_log, likes_log) maps in [0,1] for all models."""
    trending_raw: Dict[str, float] = {}
    downloads_log: Dict[str, float] = {}
    likes_log: Dict[str, float] = {}
    for m in models:
        mid = str(m.get("id"))
        likes = int(m.get("likes") or 0)
        downloads = int(m.get("downloads") or 0)
        trending_raw[mid] = likes / (downloads + 1)
        downloads_log[mid] = math.log(max(downloads, 1))
        likes_log[mid] = math.log(likes + 1)
    trending_norm = _minmax(trending_raw)
    downloads_norm = _minmax(downloads_log)
    likes_norm = _minmax(likes_log)
    return trending_norm, downloads_norm, likes_norm


def score_model(trending_norm: float, downloads_norm: float, likes_norm: float) -> float:
    return 0.4 * trending_norm + 0.35 * downloads_norm + 0.25 * likes_norm


HOST_DEFAULT = "http://localhost:8080"
MAX_PER_COMBO = 5  # for goal-mode
TOP_PER_TASK = 2  # for task-mode

# Assets directory (tests/assets)
ASSETS_DIR = Path(__file__).resolve().parent.parent / "tests" / "assets"
ASSET_IMAGE = ASSETS_DIR / "image.jpg"
ASSET_AUDIO = ASSETS_DIR / "audio.wav"
ASSET_VIDEO = ASSETS_DIR / "video.mp4"

# Simple hardcoded text prompts per task
TASK_TEXT_PROMPTS: Dict[str, str] = {
    "text-generation": "Write a short paragraph about the benefits of exercise.",
    "summarization": "Summarize: Regular exercise improves health by strengthening the heart, lowering blood pressure, and boosting mental well-being.",
    "translation": "Translate to French: The weather is nice today.",
    "question-answering": "Context: The Eiffel Tower is in Paris. Question: Where is the Eiffel Tower?",
    "table-question-answering": "Given a table of city populations, which city is largest?",
    "text-classification": "This movie was absolutely fantastic and inspiring!",
    "token-classification": "Barack Obama was born in Hawaii.",
    "zero-shot-classification": "I love hiking in the mountains.",
    "fill-mask": "The capital of France is [MASK].",
    "text-ranking": "Rank these sentences by relevance to climate change research.",
    "sentence-similarity": "Compare these sentences for similarity.",
    "feature-extraction": "Compute embeddings for this sentence about machine learning.",
    "image-to-text": "Describe the image in detail.",
    "image-text-to-text": "Question: What is the person doing in the image?",
    "text-to-image": "A cozy cabin in the snowy mountains at sunset.",
    "image-to-image": "Enhance the image quality and reduce noise.",
    "image-super-resolution": "Upscale the image to higher resolution.",
    "image-restoration": "Restore damaged areas of the image.",
    "image-to-3d": "Generate a 3D model from this image.",
    "text-to-3d": "A futuristic car with smooth curves.",
    "text-to-video": "A sunrise over the ocean with gentle waves.",
    "image-to-video": "Animate this still image into a short looping clip.",
    "automatic-speech-recognition": "Transcribe the audio sample.",
    "text-to-speech": "Welcome to our demo. This is a text-to-speech example.",
    "audio-classification": "Classify the sounds present in the audio sample.",
    "audio-to-audio": "Denoise the audio and remove background hum.",
    "text-to-audio": "Generate a short melody in a happy mood.",
    "audio-text-to-text": "Provide a summary of the spoken content.",
    "voice-activity-detection": "Detect speech segments in the audio.",
    "visual-document-retrieval": "Find matching documents by visual content.",
    "any-to-any": "General test prompt across modalities.",
}


def _read_b64(path: Path) -> str:
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    # Compose a data URL if helpful; backend expects raw base64 string, use plain
    return b64


def build_inference_inputs_for_task(task_id: str, text_fallback: str | None = None) -> Dict[str, object]:
    """Construct inputs object based on task requirements using tests/assets and hardcoded text.

    Prefers image/audio/video assets when required; includes text where applicable.
    """
    inputs: Dict[str, object] = {}
    prompt = TASK_TEXT_PROMPTS.get(task_id) or text_fallback or "Demo prompt"
    tm = build_task_modalities().get(task_id)
    if not tm:
        # default to text
        inputs["text"] = prompt
        return inputs
    req_inputs = set(tm.input)
    if "text" in req_inputs:
        inputs["text"] = prompt
    if "image" in req_inputs and ASSET_IMAGE.exists():
        inputs["image_base64"] = _read_b64(ASSET_IMAGE)
    if "audio" in req_inputs and ASSET_AUDIO.exists():
        inputs["audio_base64"] = _read_b64(ASSET_AUDIO)
    if "video" in req_inputs and ASSET_VIDEO.exists():
        inputs["video_base64"] = _read_b64(ASSET_VIDEO)
    if "document" in req_inputs:
        # No document asset available; rely on text fallback
        inputs.setdefault("text", prompt)
    # Task-specific shaping for structured inputs
    if task_id == "image-to-image":
        # Runner requires an init image and a prompt (either options.prompt or inputs.text). Use inputs.text.
        inputs.setdefault("text", prompt)
    elif task_id == "question-answering":
        # Provide explicit fields instead of a single text prompt
        inputs.pop("text", None)
        inputs["question"] = "Where is the Eiffel Tower?"
        inputs["context"] = "The Eiffel Tower is in Paris."
    elif task_id == "sentence-similarity":
        # Provide the pair explicitly
        a = "A quick brown fox jumps over the lazy dog."
        b = "A fast brown fox leaps over a lazy dog."
        inputs["text"] = a
        inputs["text_pair"] = b
    elif task_id == "table-question-answering":
        inputs.pop("text", None)
        inputs["question"] = "Which city is largest?"
        inputs["table"] = [["city", "population"], ["Alpha", "100"], ["Beta", "200"]]
    elif task_id == "text-ranking":
        inputs.pop("text", None)
        inputs["query"] = "What is the capital of France?"
        inputs["candidates"] = [
            "Berlin is the capital of Germany.",
            "Madrid is Spain's capital.",
            "Paris is the capital of France.",
        ]
    elif task_id == "zero-shot-classification":
        # Keep text, add labels
        inputs.setdefault("text", "This is a news article about technology.")
        inputs["labels"] = ["sports", "politics", "technology"]
    elif task_id == "time-series-forecasting":
        inputs.pop("text", None)
        inputs["series"] = [1.0, 1.2, 1.1, 1.3, 1.6, 1.9]

    # Always include extra_args with max_new_tokens and echo task id
    inputs["extra_args"] = {"max_new_tokens": 500, "_task": task_id}
    return inputs


def _shorten_assets(payload: Dict[str, object], max_len: int = 64) -> Dict[str, object]:
    """Return a copy of payload with base64 assets shortened for display."""
    import copy
    p = copy.deepcopy(payload)
    inputs = p.get("inputs")
    if isinstance(inputs, dict):
        for k in ("image_base64", "audio_base64", "video_base64", "document_base64"):
            v = inputs.get(k)
            if isinstance(v, str) and len(v) > max_len:
                inputs[k] = v[:max_len] + "..." + f"(len={len(v)})"
    return p


def _shorten_assets_generic(obj: object, max_len: int = 64) -> object:
    """Shorten any base64-like fields found in arbitrary dict/list structures.

    Targets keys commonly used in this project: *_base64, image/audio/video/document outputs.
    """
    def _shorten_str(s: str) -> str:
        if len(s) > max_len:
            return s[:max_len] + "..." + f"(len={len(s)})"
        return s

    if isinstance(obj, dict):
        out: Dict[str, object] = {}
        for k, v in obj.items():
            if isinstance(v, str) and (k.endswith("_base64") or k in {"image", "audio", "video", "document"}):
                out[k] = _shorten_str(v)
            else:
                out[k] = _shorten_assets_generic(v, max_len)
        return out
    elif isinstance(obj, list):
        return [_shorten_assets_generic(v, max_len) for v in obj]
    else:
        return obj


def execute_model(host: str, model: Dict[str, object], task_id: str, outln=lambda s: print(s)) -> Dict[str, object] | None:
    """Post to /api/inference for the given model and task, print brief summary."""
    # Validation-side note: ASR expects soundfile to be installed; proceed normally.
    payload = {
        "model_id": str(model.get("id")),
        "intent_id": None,
        "input_type": "text",  # primary input； backend tolerates provided inputs
        "inputs": build_inference_inputs_for_task(task_id),
        "task": task_id,
        "options": {},
    }
    url = host.rstrip("/") + "/api/inference"
    # Safety: ensure HF cache does not exceed 600GB before executing
    try:
        ensure_hf_cache_under_limit(limit_bytes=700 * (1024 ** 3))
    except Exception as _e:
        outln(f"    Warning: cache check/prune failed: {type(_e).__name__}: {_e}")
    # Show shortened payload before executing
    try:
        import json as _json
        outln("    Payload preview:")
        outln("    " + _json.dumps(_shorten_assets(payload), indent=2).replace("\n", "\n    "))
    except Exception:
        pass

    def _print_error(prefix: str, body: Dict[str, object] | None = None, status: int | None = None):
        if body is None:
            outln(f"  ✗ {prefix}")
            return
        err = body.get("error") if isinstance(body, dict) else None
        if isinstance(err, dict):
            code = err.get("code")
            message = err.get("message") or err.get("detail") or err.get("error")
            details = err.get("details")
            outln(f"  ✗ Error status={status} code={code} message={message}")
            if details:
                try:
                    import json as _json
                    outln("    details=" + _json.dumps(details, indent=2).replace("\n", "\n    "))
                except Exception:
                    outln(f"    details={details}")
        else:
            outln(f"  ✗ Error status={status} body={err}")
    try:
        if httpx is not None:
            with httpx.Client(timeout=TIMEOUT) as client:
                r = client.post(url, json=payload)
                status_code = r.status_code
                content_type = r.headers.get("content-type", "")
                data: Dict[str, object] | None = None
                try:
                    if "application/json" in content_type:
                        data = r.json()
                except Exception:
                    data = None
                if status_code != 200:
                    _print_error("HTTP error", data, status_code)
                    return
                if data is None:
                    outln(f"  ✗ Unexpected non-JSON response status={status_code}")
                    return
        else:
            # Minimal urllib fallback
            import json
            from urllib.request import Request, urlopen
            req = Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
            with urlopen(req, timeout=TIMEOUT) as resp:  # type: ignore
                status_code = getattr(resp, 'status', 200)
                raw = resp.read().decode("utf-8")
            try:
                data = json.loads(raw)
            except Exception:
                _print_error("Non-JSON response", None, status_code)
                return
            if status_code != 200:
                _print_error("HTTP error", data, status_code)
                return
        result = (data or {}).get("result") if data else None
        runtime_ms = (data or {}).get("runtime_ms") if data else None
        if result:
            # Print truncated textual result if available
            summary_keys = list(result.keys()) if isinstance(result, dict) else []
            outln(f"  ✓ Executed: runtime={runtime_ms}ms; result_keys={summary_keys}")
            try:
                import json as _json
                outln("    Response JSON:")
                safe_data = data
                # Shorten any asset-like fields in the response to avoid huge logs
                if isinstance(safe_data, dict):
                    safe_data = {
                        **safe_data,
                        "result": _shorten_assets_generic(safe_data.get("result")),
                    }
                outln("    " + _json.dumps(safe_data, indent=2).replace("\n", "\n    "))
            except Exception:
                pass
            return data
        else:
            _print_error("Execution returned error", data, 200)
            try:
                import json as _json
                outln("    Error JSON:")
                safe_err = data
                if isinstance(safe_err, dict):
                    safe_err = _shorten_assets_generic(safe_err)
                outln("    " + _json.dumps(safe_err, indent=2).replace("\n", "\n    "))
            except Exception:
                pass
            return data
    except Exception as e:
        outln(f"  ✗ Request failed: {type(e).__name__}: {e}")
        return None


def _dir_size_bytes(path: Path) -> int:
    total = 0
    try:
        for p in path.rglob('*'):
            try:
                if p.is_file():
                    total += p.stat().st_size
            except Exception:
                continue
    except Exception:
        return total
    return total


def ensure_hf_cache_under_limit(limit_bytes: int) -> None:
    """Ensure ~/.cache/huggingface/hub is under `limit_bytes` by pruning oldest entries.

    Strategy:
    - Compute total size of `~/.cache/huggingface/hub`.
    - If over limit, order immediate children by last modified time (oldest first).
    - Remove oldest directories/files until under the limit.
    """
    root = Path(os.path.expanduser('~/.cache/huggingface/hub'))
    if not root.exists():
        return
    total = _dir_size_bytes(root)
    if total <= limit_bytes:
        return
    # Gather children entries with mtime and size
    entries: List[Tuple[float, int, Path]] = []
    for child in root.iterdir():
        try:
            mtime = child.stat().st_mtime
            size = _dir_size_bytes(child)
            entries.append((mtime, size, child))
        except Exception:
            continue
    # Sort by oldest first
    entries.sort(key=lambda t: t[0])
    # Remove until under limit
    for mtime, size, child in entries:
        try:
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        except Exception:
            continue
        total -= size
        print(f"    Pruned cache entry: {child} (~{size/1e9:.2f} GB); remaining ~{total/1e9:.2f} GB")
        if total <= limit_bytes:
            break


def main() -> int:
    host = HOST_DEFAULT
    max_per = MAX_PER_COMBO
    mode = os.getenv("MODE", "goal-mode").strip().lower()

    task_modalities = build_task_modalities()
    try:
        models = fetch_models(host)
    except Exception as e:  # pragma: no cover
        print(f"ERROR: Failed to fetch models: {e}", file=sys.stderr)
        return 1

    # Exclude gated/private models before any normalization/scoring
    models = [m for m in models if not bool(m.get("gated") or False)]

    trending_norm_map, downloads_norm_map, likes_norm_map = build_normalization_maps(models)

    # Combination key: (inputs tuple, outputs tuple) -> tasks set
    combo_tasks: Dict[Tuple[Tuple[str, ...], Tuple[str, ...]], Set[str]] = {}
    for task, tm in task_modalities.items():
        key = (tuple(sorted(tm.input)), tuple(sorted(tm.output)))
        combo_tasks.setdefault(key, set()).add(task)

    # Aggregate models per combo by pipeline_tag matching tasks
    combo_models: Dict[Tuple[Tuple[str, ...], Tuple[str, ...]], List[Dict[str, object]]] = {
        k: [] for k in combo_tasks.keys()
    }
    for m in models:
        task = m.get("pipeline_tag")
        if not task or task not in task_modalities:
            continue
        tm = task_modalities[task]
        key = (tuple(sorted(tm.input)), tuple(sorted(tm.output)))
        combo_models.setdefault(key, []).append(m)

    # Prepare output file
    ts = datetime.now().strftime("%y-%m-%d-%H-%M")
    out_path = Path(f"{ts}-model-runs-{'task' if mode=='task-mode' else 'goal'}-mode.txt")
    def outln(s: str = ""):
        print(s)
        try:
            with out_path.open("a", encoding="utf-8") as f:
                f.write(s + "\n")
        except Exception:
            pass

    if mode == "task-mode":
        outln("=== Top Models per Task (MODE=task-mode) ===")
        # Group by pipeline_tag (task)
        task_models: Dict[str, List[Dict[str, object]]] = {}
        for m in models:
            pt = m.get("pipeline_tag")
            if not pt or pt not in task_modalities:
                continue
            task_models.setdefault(str(pt), []).append(m)
        for task_id in sorted(task_models.keys()):
            scored: List[Tuple[float, Dict[str, object]]] = []
            for m in task_models[task_id]:
                mid = str(m.get("id"))
                s = score_model(
                    trending_norm_map.get(mid, 0.0),
                    downloads_norm_map.get(mid, 0.0),
                    likes_norm_map.get(mid, 0.0),
                )
                scored.append((s, m))
            scored.sort(key=lambda t: t[0], reverse=True)
            top = scored[: TOP_PER_TASK]
            if not top:
                continue
            outln()
            outln(f"Task: {task_id} (models={len(scored)})")
            outln("Rank | Score | Trending | Downloads | Likes | Model ID")
            for rank, (score_val, m) in enumerate(top, start=1):
                mid = str(m.get("id"))
                likes = int(m.get("likes") or 0)
                downloads = int(m.get("downloads") or 0)
                trending = trending_norm_map.get(mid, 0.0)
                outln(
                    f"{rank:>4} | {score_val:>8.3f} | {trending:>8.3f} | {downloads:>9} | {likes:>5} | {mid}"
                )
                # Always execute
                execute_model(host, m, task_id, outln=outln)
    else:
        # goal-mode (default)
        outln("=== Top Models per Input/Output Combination (MODE=goal-mode) ===")
        for key in sorted(
            combo_tasks.keys(), key=lambda k: (len(k[0]), len(k[1]), k[0], k[1])
        ):
            inputs, outputs = key
            tasks = sorted(combo_tasks[key])
            models_for_combo = combo_models.get(key, [])
            if not models_for_combo:
                continue
            scored: List[Tuple[float, Dict[str, object]]] = []
            for m in models_for_combo:
                mid = str(m.get("id"))
                s = score_model(
                    trending_norm_map.get(mid, 0.0),
                    downloads_norm_map.get(mid, 0.0),
                    likes_norm_map.get(mid, 0.0),
                )
                scored.append((s, m))
            scored.sort(key=lambda t: t[0], reverse=True)
            top = scored[: max_per]
            outln()
            outln(
                f"Combination: inputs={list(inputs)} outputs={list(outputs)} | tasks={tasks}"
            )
            outln(
                f"Total models: {len(models_for_combo)}; showing top {len(top)} (host={host})"
            )
            outln("Rank | Score | Trending | Downloads | Likes | Model ID | Task")
            for rank, (score_val, m) in enumerate(top, start=1):
                mid = str(m.get("id"))
                likes = int(m.get("likes") or 0)
                downloads = int(m.get("downloads") or 0)
                task = str(m.get("pipeline_tag"))
                trending = trending_norm_map.get(mid, 0.0)
                outln(
                    f"{rank:>4} | {score_val:>8.3f} | {trending:>8.3f} | {downloads:>9} | {likes:>5} | {mid} | {task}"
                )
                # Always execute using the pipeline tag inferred task
                if task:
                    execute_model(host, m, task, outln=outln)

    outln("\nDone.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
