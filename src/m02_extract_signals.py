"""
m02_extract_signals.py
Extract the 3 signals needed for Stage A Auditor-Scorer training.

Signals:
  1. SAM segmentation + per-object CLIP scores
  2. CLIP Grad-CAM attention maps
  3. Object gap detection (expected vs detected)

REQUIRES: CUDA GPU (SAM + CLIP)

Run:
    python -u src/m02_extract_signals.py 2>&1 | tee logs/m02_$(date +%Y%m%d_%H%M%S).log
"""

import json
import sqlite3
from pathlib import Path
from dataclasses import dataclass, asdict

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# Local utils
from utils.prompt_parser import parse_prompt, get_color_object_pairs, ExpectedObject
from utils.clip_utils import (
    load_clip, compute_clip_score, compute_gradcam,
    classify_object, detect_color
)
from utils.sam_utils import load_sam, get_top_segments, SegmentedObject
from utils.mask_utils import encode_mask_rle


# ============================================================================
# Configuration
# ============================================================================
class Config:
    PROJECT_ROOT = Path(__file__).parent.parent
    DB_PATH = PROJECT_ROOT / "outputs" / "centralized.db"
    # Image dir (FLUX only - SDXL removed)
    IMAGES_DIR = PROJECT_ROOT / "outputs" / "m01_images_flux"
    GRADCAM_DIR = PROJECT_ROOT / "outputs" / "m02_gradcam"

    # SAM settings
    MAX_SEGMENTS = 15
    MIN_SEGMENT_AREA = 0.02  # 2% of image
    MAX_SEGMENT_AREA = 0.70  # 70% of image

    # CLIP settings
    OBJECT_MATCH_THRESHOLD = 0.55  # Min score to consider segment as matching object


# ============================================================================
# Data Structures
# ============================================================================
@dataclass
class DetectedObject:
    """An object detected in the image via SAM + CLIP."""
    segment_id: int
    bbox: tuple           # (x, y, w, h)
    area_ratio: float
    label: str            # Best matching noun from prompt
    label_score: float    # CLIP score for label
    color: str            # Detected color
    color_score: float    # Confidence in color
    prompt_score: float   # Score against full prompt phrase
    mask_rle: str = None  # Run-length encoded mask (optional, for storage)


@dataclass
class ObjectGap:
    """A mismatch between expected and detected objects."""
    object_noun: str
    issue: str            # "missing", "wrong_color", "wrong_object"
    expected: str         # What prompt asked for
    found: str            # What was detected (or None)
    segment_id: int = -1  # Which segment has the issue (-1 if missing)


@dataclass
class ImageSignals:
    """All 3 signals for a single image."""
    image_id: str
    prompt: str

    # Signal 1: Object-level detections
    objects: list[DetectedObject]

    # Signal 2: Grad-CAM path
    gradcam_path: str

    # Signal 3: Object gaps
    expected_objects: list[str]
    gaps: list[ObjectGap]

    # Summary stats
    num_objects: int
    num_gaps: int
    avg_object_score: float


# ============================================================================
# Database
# ============================================================================
def init_signals_table(conn: sqlite3.Connection) -> None:
    """Create signals table (drop and recreate for schema updates)."""
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS signals")
    cursor.execute("""
        CREATE TABLE signals (
            image_id TEXT PRIMARY KEY,
            prompt TEXT NOT NULL,
            objects_json TEXT NOT NULL,
            gradcam_path TEXT,
            expected_objects TEXT NOT NULL,
            gaps_json TEXT NOT NULL,
            num_objects INTEGER,
            num_gaps INTEGER,
            avg_object_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()


def get_images_to_process(conn: sqlite3.Connection) -> list:
    """Get all images that need signal extraction."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT i.image_id, i.generation_prompt, i.baseline_prompt, i.image_path, i.base_id, i.variation
        FROM images i
        LEFT JOIN signals s ON i.image_id = s.image_id
        WHERE s.image_id IS NULL
        ORDER BY i.base_id, i.variation
    """)
    return cursor.fetchall()


def save_signals(conn: sqlite3.Connection, signals: ImageSignals) -> None:
    """Save extracted signals to database."""
    cursor = conn.cursor()

    # Convert dataclasses to dicts for JSON
    objects_json = json.dumps([asdict(obj) for obj in signals.objects])
    gaps_json = json.dumps([asdict(gap) for gap in signals.gaps])
    expected_json = json.dumps(signals.expected_objects)

    cursor.execute("""
        INSERT OR REPLACE INTO signals
        (image_id, prompt, objects_json, gradcam_path, expected_objects,
         gaps_json, num_objects, num_gaps, avg_object_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        signals.image_id, signals.prompt, objects_json, signals.gradcam_path,
        expected_json, gaps_json, signals.num_objects, signals.num_gaps,
        signals.avg_object_score
    ))
    conn.commit()


# ============================================================================
# Signal Extraction
# ============================================================================
def extract_signal_1(
    image: Image.Image,
    prompt: str,
    expected_objects: list[ExpectedObject],
    config: Config
) -> list[DetectedObject]:
    """
    Signal 1: SAM segmentation + per-object CLIP scores.

    For each SAM segment:
      - Classify against expected object nouns
      - Detect color
      - Score against full prompt phrase
    """
    # Get SAM segments
    segments = get_top_segments(
        image,
        max_segments=config.MAX_SEGMENTS,
        min_area_ratio=config.MIN_SEGMENT_AREA,
        max_area_ratio=config.MAX_SEGMENT_AREA
    )

    if not segments:
        return []

    # Get candidate labels from prompt
    candidate_nouns = [obj.noun for obj in expected_objects]
    if not candidate_nouns:
        candidate_nouns = ["object"]  # Fallback

    detected = []
    for i, seg in enumerate(segments):
        cropped = seg.cropped_image
        if cropped is None:
            continue

        # Classify segment against expected nouns
        label, label_score = classify_object(cropped, candidate_nouns)

        # Detect color
        color, color_score = detect_color(cropped)

        # Score against relevant prompt phrase
        # Find the ExpectedObject matching this label
        matching_exp = [obj for obj in expected_objects if obj.noun == label]
        if matching_exp:
            phrase = matching_exp[0].full_phrase
        else:
            phrase = f"a {color} {label}"

        prompt_score = compute_clip_score(cropped, f"a photo of {phrase}")

        # Encode mask as RLE for precise overlay in m03
        mask_rle = encode_mask_rle(seg.mask) if seg.mask is not None else None

        detected.append(DetectedObject(
            segment_id=i,
            bbox=seg.bbox,
            area_ratio=seg.area_ratio,
            label=label,
            label_score=label_score,
            color=color,
            color_score=color_score,
            prompt_score=prompt_score,
            mask_rle=mask_rle
        ))

    return detected


def extract_signal_2(
    image: Image.Image,
    prompt: str,
    image_id: str,
    config: Config
) -> str:
    """
    Signal 2: CLIP Grad-CAM attention map.

    Returns path to saved gradcam numpy file.
    """
    config.GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

    # Compute Grad-CAM
    gradcam = compute_gradcam(image, prompt)

    # Save as numpy file
    gradcam_path = config.GRADCAM_DIR / f"{image_id}_gradcam.npy"
    np.save(str(gradcam_path), gradcam)

    return str(gradcam_path.relative_to(config.PROJECT_ROOT))


def extract_signal_3(
    expected_objects: list[ExpectedObject],
    detected_objects: list[DetectedObject],
    config: Config
) -> list[ObjectGap]:
    """
    Signal 3: Object gap detection.

    Compare expected objects (from prompt) with detected objects (from SAM+CLIP).
    """
    gaps = []

    # Track which expected objects were found
    found_nouns = set()

    for det in detected_objects:
        if det.label_score >= config.OBJECT_MATCH_THRESHOLD:
            found_nouns.add(det.label)

    # Check each expected object
    for exp in expected_objects:
        # Find matching detected objects
        matching = [d for d in detected_objects
                    if d.label == exp.noun and d.label_score >= config.OBJECT_MATCH_THRESHOLD]

        if not matching:
            # Object missing entirely
            gaps.append(ObjectGap(
                object_noun=exp.noun,
                issue="missing",
                expected=exp.full_phrase,
                found=None
            ))
            continue

        # Object found - check attributes (especially color)
        det = matching[0]  # Take best match

        # Check color
        expected_colors = [attr for attr in exp.attributes
                          if attr in {"red", "blue", "green", "yellow", "orange", "purple",
                                     "pink", "black", "white", "brown", "gray", "silver", "gold"}]

        if expected_colors and det.color not in expected_colors:
            gaps.append(ObjectGap(
                object_noun=exp.noun,
                issue="wrong_color",
                expected=expected_colors[0],
                found=det.color,
                segment_id=det.segment_id
            ))

    return gaps


def extract_all_signals(
    image: Image.Image,
    generation_prompt: str,
    baseline_prompt: str,
    variation: str,
    image_id: str,
    config: Config,
    text_header_height: int = 80
) -> ImageSignals:
    """
    Extract all 3 signals for a single image.

    CRITICAL: For variations (v1/v2/v3), gap detection compares detected objects
    against BASELINE_PROMPT (v0), not the generation_prompt.
    This ensures we detect when "purple car" differs from baseline "red car".

    Args:
        image: Full image including text header
        generation_prompt: The prompt used to generate this image
        baseline_prompt: The v0_original prompt (reference)
        variation: "v0_original", "v1_attribute", "v2_object", "v3_spatial"
        image_id: Unique identifier
        config: Configuration
        text_header_height: Pixels to crop from top (text overlay from m01)
    """
    # Crop out text header to avoid SAM segmenting text
    img_width, img_height = image.size
    if img_height > text_header_height:
        image_cropped = image.crop((0, text_header_height, img_width, img_height))
    else:
        image_cropped = image

    # For gap detection: use BASELINE prompt for variations, own prompt for v0
    # This is the KEY fix: v1/v2/v3 should be compared against v0's expectations
    if variation == "v0_original":
        gap_comparison_prompt = generation_prompt
    else:
        gap_comparison_prompt = baseline_prompt

    # Parse prompts
    # - baseline_expected: what we compare AGAINST (for gap detection)
    # - generation_expected: used for CLIP scoring (what the image tried to show)
    baseline_expected = parse_prompt(gap_comparison_prompt)
    generation_expected = parse_prompt(generation_prompt)

    expected_nouns = [obj.full_phrase for obj in baseline_expected]

    # Signal 1: SAM + CLIP object detection
    # Use generation_expected for classification (what objects the image should have)
    # But we'll compare colors against baseline_expected in gap detection
    detected = extract_signal_1(image_cropped, generation_prompt, generation_expected, config)

    # Adjust bbox coordinates to account for cropped header
    for det in detected:
        if det.bbox:
            x, y, w, h = det.bbox
            det.bbox = (x, y + text_header_height, w, h)

    # Signal 2: Grad-CAM (use baseline prompt to see where CLIP looks for expected objects)
    gradcam_path = extract_signal_2(image_cropped, baseline_prompt, image_id, config)

    # Signal 3: Object gaps - compare detected vs BASELINE expected
    # This is where we catch "purple car" vs "red car" mismatches
    gaps = extract_signal_3(baseline_expected, detected, config)

    # Compute summary stats
    avg_score = np.mean([d.prompt_score for d in detected]) if detected else 0.0

    return ImageSignals(
        image_id=image_id,
        prompt=baseline_prompt,  # Store baseline as reference
        objects=detected,
        gradcam_path=gradcam_path,
        expected_objects=expected_nouns,
        gaps=gaps,
        num_objects=len(detected),
        num_gaps=len(gaps),
        avg_object_score=float(avg_score)
    )


# ============================================================================
# Main Pipeline
# ============================================================================
def process_all_images(config: Config) -> None:
    """Extract signals for all images in database."""
    # Ensure images exist
    from utils.hf_utils import ensure_dataset_available
    if not ensure_dataset_available(config.IMAGES_DIR):
        print("ERROR: No images found. Run m01 first or download from HuggingFace.")
        return

    config.GRADCAM_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(config.DB_PATH))
    init_signals_table(conn)

    images = get_images_to_process(conn)
    if not images:
        print("No images to process (or all already processed).")
        conn.close()
        return

    print(f"Processing {len(images)} images...")
    print("Loading models...")

    # Pre-load models
    load_clip()
    load_sam()

    print("Models loaded. Extracting signals...")

    for image_id, generation_prompt, baseline_prompt, image_path, base_id, variation in tqdm(images, desc="Extracting"):
        full_path = config.PROJECT_ROOT / image_path

        if not full_path.exists():
            print(f"Image not found: {full_path}")
            continue

        try:
            image = Image.open(full_path).convert("RGB")
            signals = extract_all_signals(
                image=image,
                generation_prompt=generation_prompt,
                baseline_prompt=baseline_prompt,
                variation=variation,
                image_id=image_id,
                config=config
            )
            save_signals(conn, signals)

        except Exception as e:
            print(f"Error processing {image_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    conn.close()
    print(f"Done! Signals saved to {config.DB_PATH}")
    print(f"Grad-CAM maps saved to {config.GRADCAM_DIR}")


# ============================================================================
# Analysis / Debug
# ============================================================================
def print_signals_summary(config: Config) -> None:
    """Print summary of extracted signals."""
    conn = sqlite3.connect(str(config.DB_PATH))
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM signals")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(num_objects), AVG(num_gaps), AVG(avg_object_score) FROM signals")
    avg_obj, avg_gaps, avg_score = cursor.fetchone()

    cursor.execute("""
        SELECT image_id, num_gaps, gaps_json
        FROM signals
        WHERE num_gaps > 0
        ORDER BY num_gaps DESC
        LIMIT 5
    """)
    top_gaps = cursor.fetchall()

    conn.close()

    print(f"\n{'='*60}")
    print(f"SIGNALS SUMMARY")
    print(f"{'='*60}")
    print(f"Total images processed: {total}")
    print(f"Avg objects per image: {avg_obj:.1f}")
    print(f"Avg gaps per image: {avg_gaps:.1f}")
    print(f"Avg object score: {avg_score:.3f}")

    print(f"\nTop 5 images with most gaps:")
    for img_id, num_gaps, gaps_json in top_gaps:
        gaps = json.loads(gaps_json)
        issues = [f"{g['object_noun']}:{g['issue']}" for g in gaps]
        print(f"  {img_id}: {num_gaps} gaps - {', '.join(issues)}")


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="store_true", help="Print signals summary")
    parser.add_argument("--validate", action="store_true", help="Run validation only")
    args = parser.parse_args()

    config = Config()

    # Import validation module
    from utils.signals import run_validation

    if args.validate:
        # Validation only
        run_validation(config.DB_PATH, config.GRADCAM_DIR)
    elif args.summary:
        print_signals_summary(config)
        run_validation(config.DB_PATH, config.GRADCAM_DIR)
    else:
        process_all_images(config)
        print_signals_summary(config)
        # Run validation after extraction
        run_validation(config.DB_PATH, config.GRADCAM_DIR)
