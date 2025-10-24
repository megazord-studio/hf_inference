FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# ---- lock-aware caching ---------------------------------------------------
ARG UV_LOCK_HASH
ENV UV_LOCK_HASH=${UV_LOCK_HASH}

# ---- base OS deps --------------------------------------------------------
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl ffmpeg \
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

# ---- uv + Python 3.12 (not 3.13) -----------------------------------------
ENV PATH="/home/${USER}/.local/bin:${PATH}"
ENV UV_LINK_MODE=copy
ENV PIP_NO_CACHE_DIR=1
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
 && uv python install 3.12 \
 && uv venv --python 3.12 .venv
ENV PATH="/app/.venv/bin:${PATH}"

# ---- deps (copy files required by build backend) --------------------------
COPY --chown=${UID}:${GID} pyproject.toml uv.lock LICENSE README.md ./

# tie venv layer to the lock file
RUN echo "uv.lock=${UV_LOCK_HASH}" \
 && uv sync --frozen --link-mode=copy \
 && rm -rf /home/${USER}/.cache/uv

# ---- CUDA wheels ----------------------------------------------------------
RUN uv pip install --no-deps --link-mode=copy \
    --index-strategy unsafe-best-match \
    --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.org/simple \
    torch==2.8.0+cu128 torchvision==0.23.0+cu128 \
 && rm -rf /home/${USER}/.cache/pip /home/${USER}/.cache/uv

# ---- app code -------------------------------------------------------------
COPY --chown=${UID}:${GID} . .

# ---- sanity check ---------------------------------------------------------
RUN python -c "import uvicorn, fastapi, numpy; print('deps-ok', numpy.__version__)"

# ---- runtime --------------------------------------------------------------
ENV HF_HOME=/app/.cache/huggingface
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
