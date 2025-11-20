import logging
import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from app.routers.models import router as models_router
from app.routers.intents import router as intents_router


logger = logging.getLogger("uvicorn.error")

BASE_DIR = os.path.dirname(__file__)

app = FastAPI(title="HF Inference API", version="0.1.0")

# Include API routers
app.include_router(models_router)
app.include_router(intents_router)

# SPA catch-all (only if path not starting with /api)
@app.get("/{full_path:path}")
async def spa_catch_all(full_path: str):  # type: ignore[unused-argument]
    if full_path.startswith("api"):
        return {"detail": "Not found"}
    index_path = os.path.join(BASE_DIR, "static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"detail": "Frontend not built"}


def main() -> None:
    """CLI entry: run uvicorn with graceful KeyboardInterrupt handling.

    Using lifespan="off" avoids Starlette lifespan cancellation noise on Ctrl+C.
    If you later add startup/shutdown events, remove lifespan="off".
    """
    import uvicorn
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, lifespan="off")
    server = uvicorn.Server(config)
    try:
        server.run()
    except KeyboardInterrupt:  # clean shutdown without extra traceback
        logger.info("Received interrupt, shutting down cleanly.")
    finally:
        # Add any global cleanup here (release model handles, close pools, etc.)
        pass

__all__ = ["app", "main"]
