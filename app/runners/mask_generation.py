from app.helpers import safe_print_output

def run_mask_generation(spec, dev: str):
    safe_print_output({
        "error": "mask-generation unsupported",
        "reason": "Segment Anything models are not exposed via transformers.pipeline.",
        "hint": "Use facebook/sam... with the segment-anything library, or switch to an image-segmentation model."
    })
