"""Type definitions for the app."""

from typing import Any
from typing import Dict
from typing import Optional
from typing import TypedDict

from fastapi import UploadFile


class RunnerFiles(TypedDict, total=False):
    """Files that can be passed to a runner."""

    image: Optional[UploadFile]
    audio: Optional[UploadFile]
    video: Optional[UploadFile]


class RunnerSpec(TypedDict):
    """Specification for a runner function."""

    model_id: str
    task: str
    payload: Dict[str, Any]
    files: RunnerFiles
