from __future__ import annotations
import logging

# Logging
LOG_LEVEL = logging.INFO

# Hugging Face Hub behavior (hardcoded per guidelines)
HF_META_RETRIES = 2
HF_META_TIMEOUT_SECONDS = 10.0
HUB_LIST_TIMEOUT_SECONDS = 30.0
MODEL_ENRICH_BATCH_LIMIT = 128
HUB_LIST_LIMIT = 50000  # Explicit max limit for api.list_models calls

# Registry resource limits
REGISTRY_MAX_LOADED_MODELS = 1000
REGISTRY_MEMORY_LIMIT_MB = 32 * 1024  # 32GB logical budget

# Device overrides (always-on defaults)
DEVICE_FORCE = None  # choices: "cuda", "mps", "cpu" or None for auto
DEVICE_MAX_GPU_MEM_GB = None

