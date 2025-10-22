from __future__ import annotations

import hmac
import os
from typing import Optional

from fastapi import APIRouter
from fastapi import Form
from fastapi import Request
from fastapi.responses import RedirectResponse
from fastapi.responses import Response
from fastapi.templating import Jinja2Templates

from app.auth import LOGIN_PASSWORD
from app.auth import LOGIN_USER
from app.auth import SESSION_COOKIE_NAME
from app.auth import SESSION_TTL_SECONDS
from app.auth import _sign_session

router = APIRouter(tags=["auth"])

# Jinja2 templates (app/templates)
_templates_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "templates"
)
templates = Jinja2Templates(directory=_templates_dir)


@router.get("/login")
async def login_form(request: Request, next: Optional[str] = None) -> Response:
    if getattr(request.state, "authenticated", False):
        return RedirectResponse(next or "/", status_code=302)
    context = {
        "request": request,
        "error": None,
        "next_path": next or "/",
        "login_user": LOGIN_USER,
        "login_password_enabled": bool(LOGIN_PASSWORD),
    }
    return templates.TemplateResponse("login.html", context)


@router.post("/login")
async def login_submit(
    request: Request,
    username: str = Form(""),
    password: str = Form(""),
    next: str = Form("/"),
) -> Response:
    if getattr(request.state, "authenticated", False):
        return RedirectResponse(next or "/", status_code=302)

    if not LOGIN_PASSWORD:
        context = {
            "request": request,
            "error": "Login disabled.",
            "next_path": next,
            "login_user": LOGIN_USER,
            "login_password_enabled": False,
        }
        return templates.TemplateResponse(
            "login.html", context, status_code=403
        )

    user = (username or "").strip() or LOGIN_USER
    if user != LOGIN_USER or not hmac.compare_digest(password, LOGIN_PASSWORD):
        context = {
            "request": request,
            "error": "Invalid credentials (your fingers may be hallucinating).",
            "next_path": next,
            "login_user": LOGIN_USER,
            "login_password_enabled": True,
        }
        return templates.TemplateResponse(
            "login.html", context, status_code=401
        )

    cookie_val = _sign_session(user)
    resp = RedirectResponse(next or "/", status_code=302)
    resp.set_cookie(
        SESSION_COOKIE_NAME,
        cookie_val,
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        secure=request.url.scheme == "https",
        samesite="lax",
        path="/",
    )
    return resp


@router.get("/logout")
async def logout(next: Optional[str] = None) -> Response:
    resp = RedirectResponse(next or "/login", status_code=302)
    resp.delete_cookie(SESSION_COOKIE_NAME, path="/")
    return resp
