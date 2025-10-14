from __future__ import annotations

import hmac
from typing import Optional
from fastapi import APIRouter, Form, Request
from fastapi.responses import Response, RedirectResponse
from app.auth import (
    _sign_session,
    SESSION_COOKIE_NAME,
    SESSION_TTL_SECONDS,
    LOGIN_USER,
    LOGIN_PASSWORD,
)

router = APIRouter(tags=["auth"])

# Reworked cyberpunk / neon synthwave style login page
# --- keep everything above as-is ---

# Reworked cyberpunk / neon synthwave style login page
LOGIN_PAGE_CSS = """
:root {
  --bg:#05070b; --bg2:#08101b; --bg3:#0d1826;
  --text:#e6f3ff; --muted:#93a7c4; --line:rgba(148,163,184,.18);
  --neon:#00ffd0; --neon2:#8a2be2; --neon3:#ff5cf0; --danger:#ff4d6d;
  --glass:rgba(14,26,42,.65); --radius:20px; --focus:0 0 0 2px rgba(0,255,208,.35),0 0 0 6px rgba(255,92,240,.22);
}
*{box-sizing:border-box}
html,body{margin:0;height:100%;font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--text);}
body{
  background:
    radial-gradient(1400px 800px at 12% -10%,rgba(138,43,226,.18),transparent 60%),
    radial-gradient(1600px 900px at 88% -6%,rgba(0,255,208,.14),transparent 60%),
    linear-gradient(185deg,rgba(255,92,240,.10),transparent 40%),
    var(--bg);
  overflow:hidden;
}

/* ===== shared grid from "/" (hairlines + glow) ===== */
.page-grid,
.page-grid::before,
.page-grid::after {
  position: fixed;
  inset: 0;
  pointer-events: none;
}

.page-grid {
  z-index: 0;
  background:
    repeating-linear-gradient(0deg,   rgba(255,255,255,.10) 0 1px, transparent 1px 24px),
    repeating-linear-gradient(90deg,  rgba(255,255,255,.10) 0 1px, transparent 1px 24px);
  opacity: .35; /* stronger base visibility */
}

.page-grid::before {
  content: "";
  background:
    radial-gradient(800px 500px at 20% 0%, rgba(138,43,226,.35), transparent 60%),
    radial-gradient(900px 550px at 80% 0%, rgba(0,255,208,.30), transparent 60%);
  mix-blend-mode: screen;
  opacity: .7;
  filter: blur(0.4px);
}

.page-grid::after {
  content: "";
  background:
    linear-gradient(rgba(0,255,200,.08) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,255,200,.08) 1px, transparent 1px);
  background-size: 24px 24px, 24px 24px;
  animation: gridDrift 60s linear infinite;
  opacity: .45; /* visible neon tint */
}

@keyframes gridDrift {
  0% { background-position: 0 0, 0 0; }
  100% { background-position: 1200px 1200px, 1200px 1200px; }
}
/* ===== end shared grid ===== */

/* faint scan noise like on "/" */
body::before{
  content:"";position:fixed;inset:0;pointer-events:none;
  background-image:
    repeating-linear-gradient(0deg,rgba(255,255,255,.04) 0 1px, transparent 1px 4px);
  mix-blend-mode:overlay;opacity:.35;animation:scan 9s linear infinite;
}
@keyframes scan{0%{transform:translateY(0)}100%{transform:translateY(4px)}}

.wrap{
  position:relative;z-index:10;max-width:460px;
  margin:clamp(40px,10vh,120px) auto;padding:42px 46px 52px;
  border:1px solid var(--line);border-radius:var(--radius);
  background:
    linear-gradient(180deg,rgba(138,43,226,.12),rgba(0,0,0,.55)),
    linear-gradient(45deg,rgba(0,255,208,.10),rgba(255,92,240,.10)),
    var(--glass);
  backdrop-filter:blur(22px) saturate(140%);
  box-shadow:
    inset 0 0 0 1px rgba(0,255,208,.22),
    0 0 32px -6px rgba(0,255,208,.40),
    0 0 60px -18px rgba(255,92,240,.55);
}
.wrap::after{content:"";position:absolute;inset:0;border-radius:inherit;pointer-events:none;
  box-shadow:0 0 0 1px rgba(255,255,255,.04),0 0 0 3px rgba(0,255,208,.06),0 0 34px rgba(255,92,240,.25);
  mix-blend-mode:overlay;
}

h1{margin:0 0 10px;font-size:32px;letter-spacing:.5px;line-height:1;font-weight:900;position:relative;display:inline-block;}
.glitch{position:relative;color:var(--neon);text-shadow:0 0 14px rgba(0,255,208,.85),0 0 38px rgba(255,92,240,.32);}
.glitch::before,.glitch::after{content:attr(data-text);position:absolute;left:0;top:0;pointer-events:none;}
.glitch::before{animation:gl1 2.7s infinite linear alternate;color:var(--neon3);mix-blend-mode:screen;opacity:.55;}
.glitch::after{animation:gl2 1.8s infinite linear alternate;color:var(--neon2);mix-blend-mode:screen;opacity:.55;}
@keyframes gl1{0%{transform:translate(0,0)}20%{transform:translate(-2px,1px)}40%{transform:translate(1px,-1px)}60%{transform:translate(-1px,2px)}80%{transform:translate(2px,-1px)}100%{transform:translate(0,0)}}
@keyframes gl2{0%{transform:translate(0,0)}25%{transform:translate(2px,-1px)}50%{transform:translate(-1px,1px)}75%{transform:translate(1px,2px)}100%{transform:translate(0,0)}}
.subtitle{margin:0 0 28px;font-size:13px;letter-spacing:.35px;color:var(--muted);
  background:linear-gradient(90deg,var(--neon),var(--neon3));-webkit-background-clip:text;color:transparent;
  filter:drop-shadow(0 0 6px rgba(0,255,208,.45));
}

label{display:block;font-size:11px;letter-spacing:1.4px;font-weight:600;text-transform:uppercase;color:var(--muted);margin:0 0 6px 2px;}
.input-group{margin-bottom:22px;position:relative;}
input{width:100%;background:rgba(8,18,30,.85);border:1px solid var(--line);border-radius:14px;padding:14px 16px 13px;font-size:14px;color:var(--text);font-weight:500;letter-spacing:.3px;outline:none;box-shadow:inset 0 0 0 1px rgba(0,255,208,.08),0 0 0 0 rgba(0,255,208,.0);transition:.25s border,.25s box-shadow,.25s background;}
input:focus{border-color:var(--neon);box-shadow:var(--focus);background:#0d1f33;}
input::placeholder{color:rgba(230,243,255,.35);letter-spacing:.5px;}

button{width:100%;appearance:none;border:0;cursor:pointer;padding:16px 20px;border-radius:16px;font-weight:800;letter-spacing:.6px;font-size:15px;position:relative;isolation:isolate;overflow:hidden;color:#041014;background:linear-gradient(135deg,var(--neon),var(--neon3) 55%,var(--neon2));box-shadow:0 0 16px rgba(0,255,208,.35),0 0 42px -10px rgba(255,92,240,.85);transition:.22s transform,.22s filter,.4s box-shadow;}
button::before{content:"";position:absolute;inset:0;background:linear-gradient(120deg,rgba(255,255,255,.35),transparent 38%,transparent 62%,rgba(255,255,255,.25));mix-blend-mode:overlay;opacity:.0;transition:.4s opacity;}
button:hover{filter:brightness(1.08);transform:translateY(-2px);box-shadow:0 0 28px rgba(0,255,208,.55),0 0 60px -6px rgba(255,92,240,.9);}button:hover::before{opacity:.35}
button:active{transform:translateY(1px);filter:brightness(.95)}

.error,.info{border-radius:14px;padding:12px 16px 14px;font-size:13px;line-height:1.45;letter-spacing:.3px;margin:0 0 22px;position:relative;}
.error{background:linear-gradient(135deg,rgba(255,77,109,.15),rgba(255,77,109,.05));border:1px solid rgba(255,77,109,.55);color:#ff97ab;text-shadow:0 0 10px rgba(255,77,109,.3);} 
.info{background:linear-gradient(135deg,rgba(0,255,208,.14),rgba(138,43,226,.14));border:1px solid var(--line);color:var(--muted);text-shadow:0 0 8px rgba(0,255,208,.25);} 

.footer{margin-top:34px;font-size:11px;letter-spacing:.6px;text-align:center;color:var(--muted);opacity:.65;}
.footer a{color:var(--neon);text-decoration:none;}
.footer a:hover{text-shadow:0 0 8px rgba(0,255,208,.55);} 

@media (max-width:640px){.wrap{margin:32px 18px;padding:38px 30px 46px;}}
"""

def render_login_html(error: Optional[str] = None, next_path: str = "/") -> str:
    if not LOGIN_PASSWORD:
        # playful info when password auth disabled
        msg = "<div class='info'>Password login disabled (no INFERENCE_LOGIN_PASSWORD). Set one if you crave ceremony; otherwise your bearer token is your badge.</div>"
    else:
        msg = ""
    err_html = f"<div class='error'>{error}</div>" if error else ""
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>Login · HF Model Playground</title>
<meta name='viewport' content='width=device-width,initial-scale=1' />
<link rel='icon' type='image/x-icon' href='/static/favicon.ico' />
<style>{LOGIN_PAGE_CSS}</style></head><body>
  <!-- shared background grid (same as "/") -->
  <div class='page-grid'></div>

  <div class='wrap'>
    <h1 class='glitch' data-text='HF Inference'>HF Inference</h1>
    <p class='subtitle'>Dev Portal · authenticate and go poke some transformers</p>
    {msg}{err_html}
    <form method='post' action='/login' autocomplete='on' spellcheck='false'>
      <input type='hidden' name='next' value='{next_path}' />
      <div class='input-group'>
        <label>Username</label>
        <input name='username' placeholder='(defaults to {LOGIN_USER})' value='{LOGIN_USER}' autocomplete='username' />
      </div>
      <div class='input-group'>
        <label>Password</label>
        <input name='password' type='password' placeholder='super‑secret env var' autocomplete='current-password' />
      </div>
      <button type='submit'>BOOTSTRAP SESSION</button>
    </form>
    <div class='footer'>
      <span>Hint: creds come from env vars; treat them nicer than temp logs. · <a href='https://github.com/megazord-studio/hf_inference' target='_blank' rel='noopener'>source</a></span>
    </div>
  </div>
</body></html>"""


@router.get("/login")
async def login_form(request: Request, next: Optional[str] = None):  # type: ignore[override]
    if getattr(request.state, "authenticated", False):
        return RedirectResponse(next or "/", status_code=302)
    return Response(render_login_html(None, next or "/"), media_type="text/html")


@router.post("/login")
async def login_submit(request: Request, username: str = Form(""), password: str = Form(""), next: str = Form("/")):
    if getattr(request.state, "authenticated", False):
        return RedirectResponse(next or "/", status_code=302)
    if not LOGIN_PASSWORD:
        return Response(render_login_html("Login disabled.", next), media_type="text/html", status_code=403)
    user = username.strip() or LOGIN_USER
    if user != LOGIN_USER or not hmac.compare_digest(password, LOGIN_PASSWORD):
        return Response(render_login_html("Invalid credentials (your fingers may be hallucinating).", next), media_type="text/html", status_code=401)
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
async def logout(next: Optional[str] = None):
    resp = RedirectResponse(next or "/login", status_code=302)
    resp.delete_cookie(SESSION_COOKIE_NAME, path="/")
    return resp
