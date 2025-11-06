import logging
import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware import Middleware

import app.routes.auth_routes as auth_routes
from app.auth import SharedSecretAuthMiddleware
from app.routes import healthz as healthz_routes
from app.routes import home as home_routes
from app.routes import inference as inference_routes
from app.routes import models
from app.routes import run_form as run_form_routes
from app.throttle import ThrottleMiddleware  # added

logger = logging.getLogger("uvicorn.error")

middleware = [
    Middleware(SharedSecretAuthMiddleware, env_var="INFERENCE_SHARED_SECRET"),
    Middleware(ThrottleMiddleware),  # added
]

app = FastAPI(title="HF Inference API", version="0.1.0", middleware=middleware)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Routers
app.include_router(auth_routes.router)
app.include_router(home_routes.router)
app.include_router(healthz_routes.router)
app.include_router(inference_routes.router)
app.include_router(models.router)
app.include_router(run_form_routes.router)


def main() -> None:
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)
