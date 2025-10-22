from __future__ import annotations

from typing import Any
from typing import Dict

from fastapi import APIRouter

from app.helpers import device_str

router = APIRouter()


@router.get("/healthz")
async def healthz() -> Dict[str, Any]:
    return {"status": "ok", "device": device_str()}
