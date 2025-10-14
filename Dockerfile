# build:
#   docker buildx build --platform=linux/amd64 \
#     -t ghcr.io/megazord-studio/hf-inference:gpu-latest .

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# ---- base OS deps --------------------------------------------------------
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

# ---- non-root user -------------------------------------------------------
ARG USER=app
ARG UID=10001
ARG GID=10001
RUN groupadd -g ${GID} ${USER} \
 && useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USER} \
 && mkdir -p /app \
 && chown -R ${UID}:${GID} /app

USER ${USER}
WORKDIR /app

# ---- uv + Python 3.13 ----------------------------------------------------
ENV PATH="/home/${USER}/.local/bin:${PATH}"
ENV UV_LINK_MODE=copy
ENV PIP_NO_CACHE_DIR=1

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
 && uv python install 3.13 \
 && uv venv --python 3.13 .venv
ENV PATH="/app/.venv/bin:${PATH}"

# ---- base deps from pyproject --------------------------------------------
COPY --chown=${UID}:${GID} pyproject.toml uv.lock ./
RUN uv sync --frozen --link-mode=copy \
 && rm -rf /home/${USER}/.cache/uv || true

# ---- CUDA wheels ---------------------------------------------------------
RUN uv pip install --no-deps --link-mode=copy \
    --index-strategy unsafe-best-match \
    --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.org/simple \
    torch==2.8.0+cu128 torchvision==0.23.0+cu128 \
 && rm -rf /home/${USER}/.cache/pip /home/${USER}/.cache/uv || true

# ---- app code ------------------------------------------------------------
COPY --chown=${UID}:${GID} . .

# ---- runtime cache root for HF -------------------------------------------
ENV HF_HOME=/app/.cache/huggingface

# ---- entrypoint ----------------------------------------------------------
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
