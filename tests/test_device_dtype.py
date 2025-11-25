from app.core.device import choose_dtype


def _fake_caps(cuda: bool, mps: bool, memory_gb):
    return {
        "cuda": cuda,
        "mps": mps,
        "gpu_name": "Fake GPU" if (cuda or mps) else None,
        "memory_gb": memory_gb,
        "force_device": None,
        "max_gpu_mem_gb": None,
    }


def test_choose_dtype_cpu_always_fp32(monkeypatch):
    monkeypatch.setattr("app.core.device.device_capabilities", lambda: _fake_caps(False, False, None))
    assert choose_dtype(param_count=1_000_000_000, task="text-to-image") == "float32"
    assert choose_dtype(param_count=1_000_000_000, task="text-classification") == "float32"


def test_choose_dtype_gpu_heavy_task_prefers_fp16(monkeypatch):
    monkeypatch.setattr("app.core.device.device_capabilities", lambda: _fake_caps(True, False, 24.0))
    assert choose_dtype(param_count=1_000_000_000, task="text-to-image") == "float16"


def test_choose_dtype_gpu_small_model_prefers_fp32(monkeypatch):
    monkeypatch.setattr("app.core.device.device_capabilities", lambda: _fake_caps(True, False, 16.0))
    assert choose_dtype(param_count=10_000_000, task="text-classification") == "float32"
