from typing import Any
from typing import Dict
from typing import Optional
from typing import TypedDict

from fastapi import UploadFile


class RunnerFiles(TypedDict, total=False):
    image: Optional[UploadFile]
    audio: Optional[UploadFile]
    video: Optional[UploadFile]


class RunnerSpec(TypedDict, total=False):
    model_id: str
    task: str
    payload: Dict[str, Any]
    files: RunnerFiles
    extra_args: Dict[str, Any]  # all model kwargs go here
