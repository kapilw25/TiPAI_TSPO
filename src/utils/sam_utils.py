"""
sam_utils.py
SAM (Segment Anything Model) utilities for object segmentation.

Used for Signal 1: Object-level segmentation and scoring.
"""

import torch
import numpy as np
from PIL import Image
from dataclasses import dataclass
from pathlib import Path
# SAM imports
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


# ============================================================================
# Configuration
# ============================================================================
@dataclass
class SAMConfig:
    """SAM model configuration."""
    model_type: str = "vit_b"
    checkpoint: str = "models/sam_vit_b_01ec64.pth"
    points_per_side: int = 32
    pred_iou_thresh: float = 0.86
    stability_score_thresh: float = 0.92
    min_mask_region_area: int = 500


# ============================================================================
# Global Model Cache
# ============================================================================
_SAM_CACHE = {
    "model": None,
    "mask_generator": None,
    "device": None
}


def load_sam(config: SAMConfig = None, device: str = "cuda"):
    """Load SAM model (cached)."""
    global _SAM_CACHE

    if _SAM_CACHE["model"] is not None and _SAM_CACHE["device"] == device:
        return _SAM_CACHE["model"], _SAM_CACHE["mask_generator"]

    assert torch.cuda.is_available(), "CUDA required!"

    if config is None:
        config = SAMConfig()

    # Resolve checkpoint path
    checkpoint_path = Path(config.checkpoint)
    if not checkpoint_path.is_absolute():
        # Relative to project root
        project_root = Path(__file__).parent.parent.parent
        checkpoint_path = project_root / config.checkpoint

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint_path}")

    print(f"Loading SAM ({config.model_type}) on {device}...")
    model = sam_model_registry[config.model_type](checkpoint=str(checkpoint_path))
    model = model.to(device)

    mask_generator = SamAutomaticMaskGenerator(
        model,
        points_per_side=config.points_per_side,
        pred_iou_thresh=config.pred_iou_thresh,
        stability_score_thresh=config.stability_score_thresh,
        min_mask_region_area=config.min_mask_region_area,
    )

    _SAM_CACHE["model"] = model
    _SAM_CACHE["mask_generator"] = mask_generator
    _SAM_CACHE["device"] = device

    return model, mask_generator


# ============================================================================
# Segmentation Results
# ============================================================================
@dataclass
class SegmentedObject:
    """A segmented object from SAM."""
    mask: np.ndarray          # Binary mask (H, W)
    bbox: tuple               # (x, y, w, h)
    area: int                 # Number of pixels
    area_ratio: float         # Fraction of image
    stability_score: float    # SAM confidence
    cropped_image: Image.Image = None  # Cropped region for CLIP


def segment_image(
    image: Image.Image,
    config: SAMConfig = None,
    min_area_ratio: float = 0.01,
    max_area_ratio: float = 0.80,
    device: str = "cuda"
) -> list[SegmentedObject]:
    """
    Segment image into objects using SAM.

    Args:
        image: PIL Image
        config: SAM configuration
        min_area_ratio: Minimum segment size (fraction of image)
        max_area_ratio: Maximum segment size (fraction of image)
        device: CUDA device

    Returns:
        List of SegmentedObject, sorted by area (largest first)
    """
    _, mask_generator = load_sam(config, device)

    img_array = np.array(image.convert("RGB"))
    total_pixels = img_array.shape[0] * img_array.shape[1]

    # Generate masks
    masks = mask_generator.generate(img_array)

    if not masks:
        return []

    # Filter and convert to SegmentedObject
    objects = []
    for mask_data in masks:
        area = mask_data['area']
        area_ratio = area / total_pixels

        # Filter by size
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue

        # Get bounding box
        bbox = mask_data['bbox']  # [x, y, w, h]

        # Extract cropped region
        mask = mask_data['segmentation']
        cropped = extract_masked_region(img_array, mask)

        obj = SegmentedObject(
            mask=mask,
            bbox=tuple(bbox),
            area=area,
            area_ratio=area_ratio,
            stability_score=mask_data['stability_score'],
            cropped_image=cropped
        )
        objects.append(obj)

    # Sort by area (largest first)
    objects.sort(key=lambda x: x.area, reverse=True)

    return objects


def extract_masked_region(
    image: np.ndarray,
    mask: np.ndarray,
    padding: int = 10
) -> Image.Image:
    """
    Extract the masked region as a cropped image.

    Args:
        image: Full image array (H, W, 3)
        mask: Binary mask (H, W)
        padding: Pixels to add around bounding box

    Returns:
        PIL Image of the cropped region
    """
    # Get bounding box of mask
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        return Image.fromarray(image)

    y_indices = np.where(rows)[0]
    x_indices = np.where(cols)[0]
    y_min, y_max = y_indices[0], y_indices[-1]
    x_min, x_max = x_indices[0], x_indices[-1]

    # Add padding
    y_min = max(0, y_min - padding)
    y_max = min(image.shape[0], y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(image.shape[1], x_max + padding)

    # Crop
    cropped = image[y_min:y_max, x_min:x_max].copy()

    return Image.fromarray(cropped)


def get_top_segments(
    image: Image.Image,
    max_segments: int = 10,
    min_area_ratio: float = 0.02,
    max_area_ratio: float = 0.70,
    device: str = "cuda"
) -> list[SegmentedObject]:
    """
    Get top N largest meaningful segments from image.

    This is a convenience function that filters to the most likely
    "main objects" in the image.
    """
    segments = segment_image(
        image,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
        device=device
    )
    return segments[:max_segments]


def visualize_segments(
    image: Image.Image,
    segments: list[SegmentedObject],
    alpha: float = 0.5
) -> Image.Image:
    """
    Visualize segmentation masks overlaid on image.

    Each segment gets a random color.
    """
    import random

    img_array = np.array(image.convert("RGB")).astype(np.float32)
    overlay = img_array.copy()

    for seg in segments:
        # Random color
        color = np.array([random.randint(50, 255) for _ in range(3)], dtype=np.float32)

        # Apply color to mask region
        mask = seg.mask
        overlay[mask] = (1 - alpha) * overlay[mask] + alpha * color

    return Image.fromarray(overlay.astype(np.uint8))


def mask_to_bbox(mask: np.ndarray) -> tuple:
    """Convert binary mask to bounding box (x, y, w, h)."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        return (0, 0, 0, 0)

    y_indices = np.where(rows)[0]
    x_indices = np.where(cols)[0]

    x_min, x_max = x_indices[0], x_indices[-1]
    y_min, y_max = y_indices[0], y_indices[-1]

    return (x_min, y_min, x_max - x_min, y_max - y_min)


# ============================================================================
# Test
# ============================================================================
if __name__ == "__main__":
    print("Testing SAM utilities...")

    # Create a simple test image
    test_img = Image.new("RGB", (512, 512), color="white")

    # Draw some shapes (would need PIL.ImageDraw in practice)
    # For now, just test that SAM loads
    try:
        model, mask_gen = load_sam()
        print("SAM loaded successfully!")

        # Test segmentation on the test image
        segments = segment_image(test_img)
        print(f"Found {len(segments)} segments in test image")

    except FileNotFoundError as e:
        print(f"SAM checkpoint not found: {e}")
        print("Download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
