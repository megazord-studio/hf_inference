import warnings

# Silence noisy third-party deprecation warnings that we cannot fix locally.
# Keep the filters narrow so that in-repo DeprecationWarnings are still visible.

# transformers OwlVitProcessor deprecation (used internally by pipelines)
warnings.filterwarnings(
    "ignore",
    message="`post_process_object_detection` method is deprecated for OwlVitProcessor and will be removed in v5.",
    category=FutureWarning,
    module=r"transformers\\.models\\.owlv2\\.processing_owlv2",
)

# SWIG-related deprecations coming from llava / swig bindings
warnings.filterwarnings(
    "ignore",
    message="builtin type SwigPyPacked has no __module__ attribute",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message="builtin type SwigPyObject has no __module__ attribute",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message="builtin type swigvarlink has no __module__ attribute",
    category=DeprecationWarning,
)

