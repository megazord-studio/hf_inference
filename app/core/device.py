"""Device / acceleration selection utilities.

Gracefully handle absence of torch or GPU/MPS. Provides:
- select_device(prefer='auto') -> Optional[torch.device]
- device_capabilities() -> dict with flags & memory
- ensure_task_supported(task: Optional[str]) -> None (raises RuntimeError if acceleration required but unavailable)

Environment overrides:
FORCE_DEVICE=cuda|mps|cpu  (error if unavailable)
MAX_GPU_MEM_GB=number      (advisory; expose in capabilities)

GPU-required tasks (heuristic initial list): text-to-image, image-to-image, text-to-video,
image-to-video, image-to-3d, text-to-3d, video-generation.
"""
from __future__ import annotations
import os
import logging
from typing import Optional, Dict, Set

log = logging.getLogger("app.device")

GPU_REQUIRED_TASKS: Set[str] = {
    "text-to-image",
    "image-to-image",
    "text-to-video",
    "image-to-video",
    "image-to-3d",
    "text-to-3d",
    "video-generation",  # legacy alias
    "any-to-any",  # generalist may compose heavy tasks
}

_try_torch_done = False
_torch = None  # lazy torch import reference


def _lazy_import_torch():
    global _try_torch_done, _torch
    if _try_torch_done:
        return _torch
    try:
        import torch  # type: ignore
        _torch = torch
    except Exception as e:  # ImportError or other
        log.warning(f"Torch not available: {e}")
        _torch = None
    _try_torch_done = True
    return _torch


def device_capabilities() -> Dict[str, Optional[object]]:
    torch = _lazy_import_torch()
    caps: Dict[str, Optional[object]] = {
        "cuda": False,
        "mps": False,
        "gpu_name": None,
        "memory_gb": None,
        "force_device": os.getenv("FORCE_DEVICE"),
        "max_gpu_mem_gb": os.getenv("MAX_GPU_MEM_GB"),
    }
    if torch is None:
        return caps
    # CUDA check
    if torch.cuda.is_available():
        caps["cuda"] = True
        try:
            props = torch.cuda.get_device_properties(0)  # type: ignore
            caps["gpu_name"] = props.name
            caps["memory_gb"] = round(props.total_memory / (1024**3), 2)
        except Exception as e:
            log.debug(f"Could not read CUDA properties: {e}")
    # MPS check (Apple Silicon)
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore
            caps["mps"] = True
            if not caps.get("gpu_name"):
                caps["gpu_name"] = "Apple MPS"
    except Exception as e:
        log.debug(f"MPS check failed: {e}")
    return caps


def select_device(prefer: str = "auto") -> Optional[object]:
    """Return a torch.device or None (CPU fallback) based on preference & availability.
    Does not raise; caller may enforce requirements separately.
    """
    torch = _lazy_import_torch()
    if torch is None:
        return None
    prefer = prefer.lower()
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda:0")
    if prefer == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore
        return torch.device("mps")
    if prefer == "cpu":
        return torch.device("cpu")
    # auto logic
    if prefer == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore
            return torch.device("mps")
        return torch.device("cpu")
    # unknown prefer -> fallback auto
    return select_device("auto")


def ensure_task_supported(task: Optional[str]) -> None:
    """Raise RuntimeError if task requires GPU/MPS but none available."""
    if not task:
        return
    if task not in GPU_REQUIRED_TASKS:
        return
    caps = device_capabilities()
    if not (caps["cuda"] or caps["mps"]):
        raise RuntimeError(
            f"Task '{task}' requires CUDA or MPS acceleration; none detected. Caps={caps}. "
            "Install a GPU-enabled PyTorch build or run on Apple Silicon with MPS support."
        )


def startup_log() -> None:
    caps = device_capabilities()
    force = caps.get("force_device")
    prefer = force or "auto"
    dev = select_device(prefer)
    dev_str = str(dev) if dev else "cpu (torch missing)"
    log.info(
        "Device selection: prefer=%s resolved=%s cuda=%s mps=%s gpu_name=%s memory_gb=%s",
        prefer, dev_str, caps["cuda"], caps["mps"], caps["gpu_name"], caps["memory_gb"]
    )

__all__ = [
    "select_device",
    "device_capabilities",
    "ensure_task_supported",
    "startup_log",
    "GPU_REQUIRED_TASKS",
]
