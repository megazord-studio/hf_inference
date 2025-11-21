import logging
import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from app.routers.models import router as models_router
from app.routers.intents import router as intents_router
import app.routers.inference as inference_module
from app.core.device import startup_log


def configure_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    # Adjust third-party verbosity
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(level)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    # Application namespace
    logging.getLogger("app").setLevel(level)


# Configure logging early
configure_logging()
startup_log()

logger = logging.getLogger("app")

BASE_DIR = os.path.dirname(__file__)

app = FastAPI(title="HF Inference API", version="0.1.0")

# Include API routers
app.include_router(models_router)
app.include_router(intents_router)
app.include_router(inference_module.router)

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
