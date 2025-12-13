"""
utils package for TiPAI-TSPO.
"""

from .hf_utils import (
    push_dataset_to_hf,
    download_dataset_from_hf,
    ensure_dataset_available,
    is_dataset_available_locally,
    HF_REPO,
)

from .prompt_parser import (
    parse_prompt,
    get_object_nouns,
    get_color_object_pairs,
    compare_objects,
    ExpectedObject,
)

from .clip_utils import (
    load_clip,
    compute_clip_score,
    compute_clip_scores_batch,
    compute_gradcam,
    classify_object,
    detect_color,
)

from .sam_utils import (
    load_sam,
    segment_image,
    get_top_segments,
    extract_masked_region,
    visualize_segments,
    SegmentedObject,
    SAMConfig,
)
