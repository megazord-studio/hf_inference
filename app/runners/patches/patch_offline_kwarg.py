from typing import Any


# --- Patch: defensively strip `offload_state_dict` from ctor kwargs ---
def _patch_offload_kwarg() -> None:
    """
    Some accelerate/transformers/diffusers combos inject `offload_state_dict`
    into module __init__. Older or variant classes do not accept it.
    We monkeypatch common offenders to ignore that kwarg.
    """

    def _try_patch(qualified: str, attr: str = "__init__") -> None:
        try:
            mod_path, cls_name = qualified.rsplit(".", 1)
            mod = __import__(mod_path, fromlist=[cls_name])
            cls = getattr(mod, cls_name)
            orig = getattr(cls, attr)

            # avoid double patch
            if getattr(orig, "__name__", "") == "_patched_ignore_offload":
                return

            def _patched_ignore_offload(
                self: Any, *args: Any, **kwargs: Any
            ) -> Any:
                kwargs.pop("offload_state_dict", None)
                return orig(self, *args, **kwargs)

            _patched_ignore_offload.__name__ = "_patched_ignore_offload"
            setattr(cls, attr, _patched_ignore_offload)
        except Exception:
            pass

    # Transformers CLIP variants
    _try_patch("transformers.models.clip.modeling_clip.CLIPTextModel")
    _try_patch(
        "transformers.models.clip.modeling_clip.CLIPTextModelWithProjection"
    )
    _try_patch(
        "transformers.models.clip.modeling_clip.CLIPVisionModelWithProjection"
    )

    # OpenCLIP (some SDXL variants can use it)
    _try_patch(
        "transformers.models.open_clip.modeling_open_clip.CLIPTextModel"
    )
    _try_patch(
        "transformers.models.open_clip.modeling_open_clip.CLIPTextModelWithProjection"
    )
    _try_patch(
        "transformers.models.open_clip.modeling_open_clip.CLIPVisionModelWithProjection"
    )

    # Diffusers safety checker
    _try_patch(
        "diffusers.pipelines.stable_diffusion.safety_checker.StableDiffusionSafetyChecker"
    )
