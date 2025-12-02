from __future__ import annotations

import logging
import time
from typing import Any
from typing import Dict
from typing import Optional

from app.core.runners.base import BaseRunner

log = logging.getLogger("app.runners.multimodal.any_to_any")


class AnyToAnyRunner(BaseRunner):
    """Lightweight multimodal orchestrator.

    For now, performs modality detection and echoes a structured summary
    without invoking heavy sub-models. This turns 501 into a success path
    and provides a stable contract for future fan-out implementations.
    """

    def __init__(
        self, model_id: str, device: Any, dtype: Optional[str] = None
    ) -> None:
        super().__init__(model_id, device)
        self._loaded = False

    def unload(self) -> None:
        self._loaded = False

    def load(self) -> int:
        # No heavy components required at present
        self._loaded = True
        return 0

    def predict(
        self, inputs: Dict[str, Any], options: Dict[str, Any]
    ) -> Dict[str, Any]:
        start = time.time()
        # Detect available modalities
        mods: list[str] = []
        text = inputs.get("text")
        image_b64 = inputs.get("image_base64")
        audio_b64 = inputs.get("audio_base64")
        video_b64 = inputs.get("video_base64")
        if text:
            mods.append("text")
        if image_b64:
            mods.append("image")
        if audio_b64:
            mods.append("audio")
        if video_b64:
            mods.append("video")
        # Minimal structured output; downstream can expand
        output: Dict[str, Any] = {
            "modalities": mods,
            "text": text or "",
        }
        # Registry will wrap with metadata and timing; we return payload only
        _ = int((time.time() - start) * 1000)
        return output

    def predict_stream(
        self, inputs: Dict[str, Any], options: Dict[str, Any]
    ) -> Any:
        # Streaming not supported; return a single synthesized chunk
        yield {"event": "error", "data": "streaming_not_supported"}
